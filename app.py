import gradio as gr
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
import os

chat_histories = {}

def load_vectorstore():
    embedding_fn = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    vectorstore = Chroma(persist_directory="vectorstore", embedding_function=embedding_fn)
    return vectorstore.as_retriever(search_kwargs={"k": 6}), vectorstore

def load_llm():
    api_key = "AIzaSyBmk_5xiADBgdBQuNYavK_HPKruT1xBuTQ"
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20", google_api_key=api_key, temperature=0.3)

retriever, vectorstore = load_vectorstore()
llm = load_llm()

def get_all_papers():
    collection = vectorstore._collection
    all_docs = collection.get(include=['metadatas'])
    sources = set()
    for metadata in all_docs['metadatas']:
        if 'source' in metadata:
            sources.add(metadata['source'])
    return sorted(list(sources))

indexed_papers = get_all_papers()

def generate_multi_queries(question):
    template = """You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant medical research documents. 
    Provide these alternative questions separated by newlines. Original question: {question}"""
    prompt = ChatPromptTemplate.from_template(template)
    generate_queries = prompt | llm | StrOutputParser() | (lambda x: x.split("\n"))
    return generate_queries.invoke({"question": question})

def get_unique_union(documents):
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    unique_docs = list(set(flattened_docs))
    return [loads(doc) for doc in unique_docs]

def retrieve_documents(question):
    queries = generate_multi_queries(question)
    all_docs = []
    for query in queries:
        if query.strip():
            docs = retriever.get_relevant_documents(query.strip())
            all_docs.append(docs)
    return get_unique_union(all_docs)

def update_summary(session_id, new_question, new_answer):
    if session_id not in chat_histories:
        chat_histories[session_id] = {"messages": [], "summary": ""}
    
    session_data = chat_histories[session_id]
    
    if session_data["summary"]:
        summary_prompt = f"""Given the existing conversation summary and the new exchange, 
        generate a new summary. Maintain relevant information.
        
        Existing summary: {session_data["summary"]}
        New exchange:
        User: {new_question}
        Assistant: {new_answer}
        
        Generate the updated summary:"""
    else:
        summary_prompt = f"""Summarize this conversation:
        User: {new_question}
        Assistant: {new_answer}"""
    
    new_summary = llm.invoke(summary_prompt)
    session_data["summary"] = new_summary.content

def chat(session_id, message, history):
    if not session_id or not session_id.strip():
        return "Please enter a valid Session ID first."
    
    if session_id not in chat_histories:
        chat_histories[session_id] = {"messages": [], "summary": ""}
    
    session_data = chat_histories[session_id]
    
    docs = retrieve_documents(message)
    context = "\n\n".join([doc.page_content for doc in docs[:5]])
    
    template = """You are a medical research assistant. Answer based on the research context and conversation history.

Conversation Summary: {summary}

Research Context: {context}

Question: {question}

Provide a detailed answer based on the research papers. If the answer is not in the context, say so."""

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    answer = chain.invoke({
        "summary": session_data["summary"] if session_data["summary"] else "No previous conversation.",
        "context": context,
        "question": message
    })
    
    sources = "\n\n📚 **Sources:**\n"
    for i, doc in enumerate(docs[:3], 1):
        source_name = doc.metadata.get('source', 'Unknown')
        content_preview = doc.page_content[:200].replace('\n', ' ')
        sources += f"\n**{i}.** {source_name}\n_{content_preview}..._\n"
    
    full_response = answer + "\n\n" + sources
    
    session_data["messages"].append((message, answer))
    update_summary(session_id, message, answer)
    
    return full_response

def get_summary(session_id):
    if session_id and session_id.strip() and session_id in chat_histories and chat_histories[session_id]["summary"]:
        return chat_histories[session_id]["summary"]
    return "No conversation summary yet. Start chatting to generate a summary."

def clear_history(session_id):
    if session_id and session_id.strip() and session_id in chat_histories:
        chat_histories[session_id] = {"messages": [], "summary": ""}
        return "History cleared!", "", "History cleared successfully!"
    return "", "", "Please enter a valid Session ID first."

def view_paper_details(paper_name):
    if not paper_name:
        return "Please select a paper from the dropdown."
    
    collection = vectorstore._collection
    all_docs = collection.get(include=['documents', 'metadatas'])
    
    paper_chunks = []
    for i, metadata in enumerate(all_docs['metadatas']):
        if metadata.get('source') == paper_name:
            paper_chunks.append({
                'content': all_docs['documents'][i],
                'page': metadata.get('page', 'Unknown')
            })
    
    if not paper_chunks:
        return f"No content found for {paper_name}"
    
    result = f"# 📄 {paper_name}\n\n"
    result += f"**Total Chunks:** {len(paper_chunks)}\n\n"
    result += "## Preview (First 3 chunks):\n\n"
    
    for i, chunk in enumerate(paper_chunks[:3], 1):
        result += f"**Chunk {i} (Page {chunk['page']}):**\n"
        result += f"{chunk['content'][:500]}...\n\n"
        result += "---\n\n"
    
    return result

with gr.Blocks(title="Medical Research Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏥 Medical Research Assistant")
    gr.Markdown("Ask questions about cancer, diabetes, and cardiology research papers")
    
    with gr.Row():
        with gr.Column(scale=3):
            session_input = gr.Textbox(
                label="Session ID",
                placeholder="Enter your unique ID (e.g., user123)",
                info="This ID keeps your chat history separate from others"
            )
            msg = gr.Textbox(
                label="Your Question",
                placeholder="Ask about medical research...",
                lines=2
            )
            
            with gr.Row():
                submit = gr.Button("Send", variant="primary", scale=2)
                clear = gr.Button("Clear History", scale=1)
            
            chatbot = gr.Chatbot(height=500, show_label=False)
        
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Conversation Summary")
            summary_box = gr.Textbox(
                label="",
                lines=12,
                interactive=False,
                show_label=False
            )
            refresh_summary = gr.Button("Refresh Summary", variant="secondary")
            
            gr.Markdown("### 📚 Indexed Papers")
            paper_dropdown = gr.Dropdown(
                choices=indexed_papers,
                label="Select a paper to view",
                interactive=True,
                value=None
            )
            view_paper_btn = gr.Button("View Paper Details", variant="secondary")
            
            paper_details = gr.Markdown(label="Paper Details")
            
            gr.Markdown(f"**Total Papers:** {len(indexed_papers)}")
    
    def respond(session_id, message, chat_history):
        bot_message = chat(session_id, message, chat_history)
        chat_history.append((message, bot_message))
        return "", chat_history, get_summary(session_id)
    
    msg.submit(respond, [session_input, msg, chatbot], [msg, chatbot, summary_box])
    submit.click(respond, [session_input, msg, chatbot], [msg, chatbot, summary_box])
    
    clear.click(clear_history, [session_input], [msg, chatbot, summary_box])
    refresh_summary.click(get_summary, [session_input], [summary_box])
    
    view_paper_btn.click(view_paper_details, [paper_dropdown], [paper_details])

demo.launch()
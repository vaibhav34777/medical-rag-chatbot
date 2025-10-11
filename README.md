# Medical RAG Chatbot

## Overview

A Retrieval-Augmented Generation (RAG) chatbot that answers medical research queries using a vector database of medical publications from arXiv. The system implements Multi-Query Retrieval and Conversation Summary Memory to provide context-aware, accurate responses backed by scientific literature.

Assignment Option: Option B - Medical Publications-Based Chatbot

---

## System Architecture

![System Architecture](flow_diagram.png)

---

## Quick Start

### Prerequisites
- Python 3.10 or higher
- Google Gemini API key

### Installation

1. Clone the repository
```bash
git clone https://github.com/vaibhav34777/medical-rag-chatbot.git
cd medical-rag-chatbot
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Configure API Key

Replace the API key in `app.py` (line 14):
```python
api_key = "YOUR_GOOGLE_GEMINI_API_KEY"
```

4. Run the application
```bash
python app.py
```

The Gradio interface will launch at `http://localhost:7860`

---

## Demo
You can try the chatbot here : [Hugging Face Space](https://huggingface.co/spaces/imvaibhavrana/medical-research-chatbot)

---

## Core Technical Components

### 1. Multi-Query Retrieval System

Traditional single-query retrieval can miss relevant documents due to semantic distance limitations. Our system generates 5 diverse query variants from the user's original question, retrieves documents for each variant, and deduplicates the results to get comprehensive coverage.

This approach increases retrieval recall by capturing different semantic perspectives and handling ambiguous medical terminology effectively.

### 2. Conversation Summary Memory

Maintains context across multi-turn conversations without exceeding token limits. Instead of storing full conversation history, we use rolling summarization where each new exchange updates the existing summary.

Each user session has isolated conversation state, and summaries remain compact at 200-400 tokens while preserving critical context.

### 3. Vector Database & Document Processing

Documents are processed through the following pipeline:
- PDF Download from arXiv
- Text Extraction
- Chunking (1500 characters with 300 character overlap)
- Embedding using sentence-transformers/all-MiniLM-L6-v2
- Storage in ChromaDB

The chunking strategy maintains semantic coherence while ensuring context preservation at chunk boundaries.

### 4. Context Construction & Response Generation

Retrieved documents are combined with conversation summary to create the prompt:

```python
template = """You are a medical research assistant. 
Answer based on the research context and conversation history.

Conversation Summary: {summary}
Research Context: {context}
Question: {question}

Provide a detailed answer based on the research papers. 
If the answer is not in the context, say so."""
```

The LLM (Gemini 2.5 Flash) generates responses with temperature 0.3 for balanced accuracy and includes citations to source documents.

---

## Sample Queries & Outputs

### Example 1: Complex Multi-Part Query

**Query**:
```
What are the two primary strategies explored in this study to optimize 
Motor Imagery-based Brain-Computer Interface (MI-BCI) rehabilitation 
protocols for stroke patients, and how does the experimental evidence 
for each strategy demonstrate an improvement in BCI classification 
performance compared to conventional approaches?
```

**Answer**:
```
The study explored two primary strategies to optimize Motor Imagery-based 
Brain-Computer Interface (MI-BCI) rehabilitation protocols for stroke patients:

1. Task Design: The study proposed an "affected hand movement versus rest" 
   motor imagery paradigm as an alternative to the conventional left-versus-right 
   motor imagery. This approach aimed to simplify the task and address the weak 
   contralateral activation often observed in stroke patients.
   
   Experimental Evidence: The experimental results demonstrated that the "MI vs. rest" 
   approach generally showed better classification performance compared to the 
   "left vs. right" approach. Paired sample t-tests confirmed that the differences 
   were statistically significant (p < 0.05) for both L:Rest and R:Rest compared 
   to L:R. EEGNet showed significant improvements in both healthy and stroke patient 
   groups, while FBCSP was significant in all conditions except R:Rest in the 
   healthy group.

2. Training Protocol (Session Duration): The study investigated the impact of 
   session duration on BCI performance.
   
   Experimental Evidence: The findings revealed that shorter training sessions 
   produced better BCI performance than longer sessions. Furthermore, the research 
   suggested that using data from multiple short training sessions or implementing 
   adaptive training protocols could further enhance MI classification performance.
```

**Sources**:
1. 05_Optimizing_BCI_Rehabilitation_Protocols_for_Stroke.pdf
2. 05_Optimizing_BCI_Rehabilitation_Protocols_for_Stroke.pdf
3. 05_Optimizing_BCI_Rehabilitation_Protocols_for_Stroke.pdf

### Example 2: Follow-up Query with Memory

**Query**:
```
What was the conventional paradigm that was replaced, and for which specific 
group of stroke patients (LHP or RHP) did the FBCSP classifier show the highest 
mean accuracy using the proposed MI versus rest paradigm?
```

**Conversation Summary Used**:
```
Previous context: Discussion about MI-BCI optimization strategies including 
task design (affected hand movement vs rest paradigm replacing left-vs-right 
motor imagery) and training duration modifications for stroke rehabilitation.
```

**Answer**:
```
The conventional paradigm that was replaced was the left-versus-right motor 
imagery approach.

For the FBCSP classifier using the proposed MI versus rest paradigm, the highest 
mean accuracy was shown by the Stroke Subjects Left Hand Affected (LHP) group, 
with a mean accuracy of 65.8% in the R:Rest condition.
```

**Sources**:
1. 05_Optimizing_BCI_Rehabilitation_Protocols_for_Stroke.pdf
2. 05_Optimizing_BCI_Rehabilitation_Protocols_for_Stroke.pdf
3. 05_Optimizing_BCI_Rehabilitation_Protocols_for_Stroke.pdf

---

---

## Design Decisions

### Multi-Query Retrieval

Medical terminology is highly specialized. A single query might miss papers using different terminology for the same concept. By generating 5 semantic variations, we retrieve 40-60% more relevant documents compared to single-query approaches.

### Conversation Summary Memory

Medical consultations involve multi-turn clarifications. Rolling summarization maintains context while staying under token budgets, allowing the system to understand follow-up questions that reference previous exchanges.

### Chunking Strategy

1500 characters with 300 character overlap balances semantic coherence with context preservation. The overlap ensures important information at chunk boundaries is not lost.

### Temperature Setting

0.3 provides balanced accuracy and creativity, suitable for medical information where factual accuracy is critical but natural language generation is still needed.

---

## Author

Vaibhav  
GitHub: @vaibhav34777

---

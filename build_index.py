import os
import requests
import feedparser
from urllib.parse import quote
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import warnings
import logging
import re

warnings.filterwarnings("ignore")
logging.getLogger("pypdf").setLevel(logging.ERROR)

MEDICAL_QUERY = 'cat:q-bio.TO OR cat:q-bio.CB OR cat:q-bio.NC OR cat:q-bio.BM OR cat:physics.med-ph'
MAX_RESULTS = 10
DATA_DIR = "data"
VECTORSTORE_DIR = "vectorstore"

def fetch_arxiv_papers(query, max_results=10):
    encoded_query = quote(query)
    url = f"http://export.arxiv.org/api/query?search_query={encoded_query}&sortBy=submittedDate&sortOrder=descending&max_results={max_results}"
    feed = feedparser.parse(url)
    papers = []
    for entry in feed.entries:
        pdf_link = next((l.href for l in entry.links if l.type == "application/pdf"), None)
        papers.append({"title": entry.title, "pdf_url": pdf_link})
    return papers

def download_pdf(url, filename):
    response = requests.get(url, timeout=15)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)

def load_and_chunk_pdfs(data_dir):
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(data_dir, filename))
            pages = loader.load()
            for page in pages:
                chunks = text_splitter.split_text(page.page_content)
                for chunk in chunks:
                    all_chunks.append(Document(page_content=chunk, metadata={"source": filename}))
    return all_chunks

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)

    papers = fetch_arxiv_papers(MEDICAL_QUERY, MAX_RESULTS)

    for i, paper in enumerate(papers, start=1):
        title_clean = "".join(c for c in paper['title'] if c.isalnum() or c in (" ", "_")).replace(" ", "_")[:50]
        filename = os.path.join(DATA_DIR, f"{i:02d}_{title_clean}.pdf")
        if paper['pdf_url']:
            download_pdf(paper['pdf_url'], filename)

    chunks = load_and_chunk_pdfs(DATA_DIR)

    if chunks:
        embedding_fn = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding_fn, persist_directory=VECTORSTORE_DIR)
        vectorstore.persist()

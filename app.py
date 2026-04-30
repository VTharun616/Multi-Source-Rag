import os
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# -----------------------------
# LOAD ENV
# -----------------------------
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("API key not loaded. Check .env file")

os.environ["GOOGLE_API_KEY"] = api_key

# -----------------------------
# LLM
# -----------------------------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# -----------------------------
# EMBEDDINGS
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# LOAD DATA
# -----------------------------
pdf_docs = PyPDFLoader("raag.pdf").load()

web_docs = WebBaseLoader(
    "https://www.ibm.com/think/topics/deep-learning"
).load()

# -----------------------------
# CHUNKING (IMPROVED)
# -----------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200
)

pdf_chunks = splitter.split_documents(pdf_docs)
web_chunks = splitter.split_documents(web_docs)

# -----------------------------
# VECTOR DB
# -----------------------------
pdf_db = FAISS.from_documents(pdf_chunks, embeddings)
web_db = FAISS.from_documents(web_chunks, embeddings)

pdf_retriever = pdf_db.as_retriever(search_kwargs={"k": 5})
web_retriever = web_db.as_retriever(search_kwargs={"k": 5})

# -----------------------------
# ROUTER + FIX DL CASE
# -----------------------------
def route_query(query):
    query_lower = query.lower()

    if query_lower == "dl":
        query = "deep learning"

    return query

# -----------------------------
# GET DOCS (HYBRID SEARCH)
# -----------------------------
def get_docs(query):
    pdf_docs = pdf_retriever.invoke(query)
    web_docs = web_retriever.invoke(query)
    return pdf_docs + web_docs

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("📚 Multi-Source RAG Chatbot (Improved)")

query = st.text_input("Ask your question")

if query:

    processed_query = route_query(query)

    docs = get_docs(processed_query)

    if not docs:
        st.write("No relevant context found. Try rephrasing.")
        st.stop()

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are an expert AI tutor.

Rules:
- Answer in 3–5 clear lines
- Give explanation + simple understanding
- Use ONLY context
- If not in context, say: "I don't know from the provided data"

Context:
{context}

Question:
{query}

Answer:
"""

    try:
        response = llm.invoke(prompt)
        st.write(response.content)

    except Exception as e:
        st.error(str(e))
# 🤖 Multi-Source RAG Chatbot
A **Retrieval-Augmented Generation (RAG)** based chatbot that answers questions using multiple data sources like **PDF documents and web pages** using LLMs.

Streamlit:
https://multi-source-rag-4hlqntxcf77trcqcsqdps2.streamlit.app/

HuggingFace:
https://huggingface.co/spaces/VTharun616/Multi-Source-Rag

## 1. Project Overview
This project is an AI-powered chatbot that:
- Takes user questions in chat format
- Retrieves relevant information from:
  - 📄 PDF document
  - 🌐 Web page content
- Uses embeddings + FAISS for semantic search
- Generates human-like answers using Google Gemini LLM

## 2. Tech Stack
- Streamlit (UI)
- LangChain
- Google Gemini API (ChatGoogleGenerativeAI)
- HuggingFace Embeddings
- FAISS (Vector Database)
- PyPDF
- BeautifulSoup / Web scraping
- Sentence Transformers

## 3. How It Works
1. Load PDF and web content  
2. Split text into chunks  
3. Convert chunks into embeddings  
4. Store in FAISS vector database  
5. Retrieve relevant chunks for user query  
6. Send context + question to Gemini LLM  
7. Display chatbot response  

## 4. Features
- Chatbot-style UI (like ChatGPT)
- Multi-source knowledge (PDF + Website)
- Context-aware answers
- Session memory (chat history)
- Fast semantic search using FAISS

import os
import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="Multi-Source RAG Chatbot",
    page_icon="🤖"
)

st.title("🤖 Multi-Source RAG Chatbot")
st.write("Ask questions from PDF + Website content")

# -----------------------------------
# API KEY
# -----------------------------------
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("GOOGLE_API_KEY not found. Add it in Hugging Face Secrets.")
    st.stop()

# -----------------------------------
# LLM
# -----------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key
)

# -----------------------------------
# EMBEDDINGS
# -----------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------------
# CACHE VECTOR DATABASE
# -----------------------------------
@st.cache_resource
def load_vector_db():
    # Load PDF
    pdf_docs = PyPDFLoader("raag.pdf").load()

    # Load Website
    web_docs = WebBaseLoader(
        "https://www.ibm.com/think/topics/deep-learning"
    ).load()

    # Text Splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200
    )

    pdf_chunks = splitter.split_documents(pdf_docs)
    web_chunks = splitter.split_documents(web_docs)

    all_chunks = pdf_chunks + web_chunks

    # Create Vector DB
    db = FAISS.from_documents(all_chunks, embeddings)

    return db

db = load_vector_db()

retriever = db.as_retriever(
    search_kwargs={"k": 5}
)

# -----------------------------------
# CHAT HISTORY
# -----------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------------------
# CHAT INPUT
# -----------------------------------
user_question = st.chat_input("Ask your question here...")

if user_question:
    # Save and show user message
    st.session_state.messages.append(
        {"role": "user", "content": user_question}
    )

    with st.chat_message("user"):
        st.markdown(user_question)

    # Retrieve relevant documents
    docs = retriever.get_relevant_documents(user_question)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
    Answer the question using the context below.

    Context:
    {context}

    Question:
    {user_question}
    """

    # LLM response
    response = llm.invoke(prompt)
    answer = response.content

    # Save and show assistant response
    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

# app.py - Streamlit interface for the LangChain RAG pipeline using Ollama

"""Simple Streamlit app that lets the user upload a PDF, builds a vector store with HuggingFace embeddings,
and answers questions using a local LLM via Ollama.

The app mirrors the logic in `rag_pipeline.py` but provides an interactive UI.
"""

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaLLM


def format_docs(docs):
    """Utility to combine document chunks into a single string for the prompt."""
    return "\n\n".join(doc.page_content for doc in docs)

st.set_page_config(page_title="RAG with Ollama", layout="centered")
st.title("📄 PDF Question‑Answering with Ollama")
st.write(
    "Upload a PDF, then ask questions about its content. "
    "The system uses a local LLM (via Ollama) and sentence‑transformer embeddings."
)

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
if uploaded_file is not None:
    # Save uploaded file to a temporary location
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    # Load and split the document
    loader = PyPDFLoader("uploaded.pdf")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    st.success(f"Document split into {len(chunks)} chunks.")

    # Build vector store
    embedding_model = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    st.success("Vector store created.")

    # Prepare the retrieval‑augmented generation chain
    retriever = vectorstore.as_retriever()
    template = """Answer the question based only on the following context:{context}\n\nquestion: {question}"""
    prompt = PromptTemplate.from_template(template)
    import os
    model_name = os.getenv("OLLAMA_MODEL", "tinyllama")
    llm = OllamaLLM(model=model_name)
    qa_chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm)

    # Question input
    query = st.text_input("Ask a question about the PDF")
    if query:
        with st.spinner("Generating answer…"):
            answer = qa_chain.invoke(query)
        st.subheader("Answer")
        st.write(answer)
else:
    st.info("Upload a PDF to get started.")
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

loader = PyPDFLoader("NSEL_Accountancy_Project_Merged_20_25_Handwritten_Equivalent.pdf")
documents = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Split the document into {len(chunks)} chunks.")

embedding_model = HuggingFaceEmbeddings()

vectorstore = FAISS.from_documents(
    chunks, 
    embedding_model
)

print("Vectorstore created successfully.")

query = "What is the purpose of the project?"
similar_docs = vectorstore.similarity_search(query)

print(f"Query: {query}")
print(f"Similar documents: {similar_docs[0].page_content}")

template = """Answer the question based only on the following context:{context}

question: {question}
"""
import os
model_name = os.getenv("OLLAMA_MODEL", "mistral")
prompt = PromptTemplate.from_template(template)
llm = OllamaLLM(model=model_name)
retriever = vectorstore.as_retriever()
# Retrieve relevant docs and format context
relevant_docs = retriever.invoke(query)
context = format_docs(relevant_docs)
try:
    answer = (prompt | llm).invoke({"context": context, "question": query})
    print("Final answer:", answer)
except Exception as e:
    print("LLM generation failed:", e)
    print("Context preview:\n", context[:500])
print("Final answer:", answer)
# Duplicate print removed
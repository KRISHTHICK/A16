from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
import os
import tempfile
import streamlit as st

# Title
st.title("📄 RAG Chatbot using Ollama (Local & No API Key)")

# Upload PDF
uploaded_file = st.file_uploader("Upload a document (PDF)", type=["pdf"])

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Initialize vectorstore and llm outside the conditional block
vectorstore = None
llm = Ollama(model="gemma")

if uploaded_file:
    # Save uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_pdf_path = tmp_file.name

    # Load & Split PDF
    loader = PyPDFLoader(tmp_pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    # Embedding + Vector Store
    embeddings = OllamaEmbeddings(model="gemma")
    vectorstore = Chroma.from_documents(docs, embedding=embeddings)

    # Remove temporary file
    os.remove(tmp_pdf_path)

# User Input
user_question = st.text_input("Ask a question:")

if user_question:
    st.markdown("### 🤖 Response")

    # Option 1: Answer using RAG if vectorstore is available
    st.subheader("Answer using Document (RAG)")
    if vectorstore:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            return_source_documents=True
        )
        rag_result = qa_chain.invoke({"query": user_question})
        rag_answer = rag_result["result"]
        st.write(rag_answer)
        st.session_state.history.append(("You", user_question))
        st.session_state.history.append(("Bot (RAG)", rag_answer))
    else:
        st.info("Please upload a PDF document to use the document-based answer.")

    # Option 2: Answer without using RAG (just the base LLM)
    st.subheader("Answer without Document")
    base_llm_answer = llm.invoke(user_question)
    st.write(base_llm_answer)
    st.session_state.history.append(("You", user_question))
    st.session_state.history.append(("Bot (No RAG)", base_llm_answer))

# Display chat history
if st.session_state.history:
    st.markdown("### 💬 Chat History")
    for speaker, message in st.session_state.history:
        st.write(f"**{speaker}:** {message}")

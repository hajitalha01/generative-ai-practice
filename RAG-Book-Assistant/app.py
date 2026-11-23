
import streamlit as st
import tempfile
import os
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


st.set_page_config(page_title="RAG Book Assistant")

st.title("📚 RAG Book Assistant")
st.write("Upload a PDF and ask questions from your document.")


# Local embedding model
@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="models/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


# Local LLM
@st.cache_resource
def load_llm():

    torch.set_num_threads(min(4, os.cpu_count() or 4))

    model_path = "models/TinyLlama-1.1B-Chat"

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        low_cpu_mem_usage=True
    )

    model.eval()

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128,
        do_sample=False,
        num_beams=1,
        return_full_text=False
    )

    return HuggingFacePipeline(pipeline=pipe)


# Prompt
prompt = PromptTemplate.from_template(
    """<|system|>
You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: I could not find the answer in the document.
</s>
<|user|>
Context:
{context}

Question:
{question}
</s>
<|assistant|>
"""
)


# PDF upload
st.subheader("Upload Your PDF")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)


# Create database button
if uploaded_file:

    st.success(f"Selected PDF: {uploaded_file.name}")

    if st.button("Create Vector Database"):

        with st.spinner("Loading and processing PDF..."):

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as tmp_file:

                tmp_file.write(uploaded_file.getvalue())
                file_path = tmp_file.name

            try:

                # Load PDF
                loader = PyPDFLoader(file_path)
                docs = loader.load()

                st.write(f"Total pages: {len(docs)}")

                # Split into chunks
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )

                chunks = splitter.split_documents(docs)

                st.write(f"Total chunks: {len(chunks)}")

                # Load embeddings
                embeddings = load_embeddings()

                # Remove old database before replacing
                if os.path.exists("chroma_db"):
                    shutil.rmtree("chroma_db")

                # Create new Chroma database
                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory="chroma_db"
                )

                st.session_state["db_created"] = True
                st.session_state["pdf_name"] = uploaded_file.name

                st.success("Vector database created successfully!")

            except Exception as e:
                st.error(f"Error: {e}")

            finally:
                if os.path.exists(file_path):
                    os.remove(file_path)


# Question answering section
if st.session_state.get("db_created") and os.path.exists("chroma_db"):

    st.divider()

    st.subheader("Ask Questions From Your PDF")

    query = st.text_input("Enter your question")

    if st.button("Get Answer", disabled=not query.strip()):

        with st.spinner("Searching and generating answer..."):

            try:

                embeddings = load_embeddings()

                vectorstore = Chroma(
                    persist_directory="chroma_db",
                    embedding_function=embeddings
                )

                retriever = vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 2}
                )

                # Retrieve relevant chunks
                docs = retriever.invoke(query)

                context = "\n\n".join(
                    [doc.page_content for doc in docs]
                )

                # Create prompt
                final_prompt = prompt.format(
                    context=context,
                    question=query
                )

                # Generate answer
                llm = load_llm()

                response = llm.invoke(final_prompt)

                st.subheader("🤖 AI Answer")
                st.write(response)

            except Exception as e:
                st.error(f"Error generating answer: {e}")

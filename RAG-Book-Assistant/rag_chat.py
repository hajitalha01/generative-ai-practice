
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os

# Local embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="models/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Load Chroma database
if not os.path.exists("chroma_db"):
    raise FileNotFoundError("Chroma database not found. Run create_database.py first.")

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

# Retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)

# Local LLM model path
model_path = "models/TinyLlama-1.1B-Chat"

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True
)

# Text generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.2,
    top_p=0.9,
    return_full_text=False
)

llm = HuggingFacePipeline(
    pipeline=pipe
)

print("Local LLM loaded successfully!")

# Prompt template
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

print("RAG system created!")
print("Press 0 to exit.")

# Chat loop
while True:

    query = input("\nYou: ")

    if query == "0":
        break

    # Retrieve relevant documents
    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    # Generate response
    final_prompt = prompt.format(
        context=context,
        question=query
    )

    response = llm.invoke(final_prompt)

    print("\nAI:", response)

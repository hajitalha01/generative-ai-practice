
import torch
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

# 1. Model information
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# 2. Local download directory
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "TinyLlama-1.1B-Chat"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("Downloading model from Hugging Face...")

# 3. Download model automatically
snapshot_download(
    repo_id=MODEL_ID,
    local_dir=str(MODEL_DIR)
)

print("Model downloaded successfully!")
print("Saved at:", MODEL_DIR)

# 4. Load model locally
print("\nLoading local model...")

pipe = pipeline(
    "text-generation",
    model=str(MODEL_DIR),
    tokenizer=str(MODEL_DIR),
    device=-1,
    torch_dtype=torch.float32,
    max_new_tokens=150,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    return_full_text=False,
    pad_token_id=0
)

print("Model loaded successfully!")

# 5. LangChain HuggingFace Pipeline
llm = HuggingFacePipeline(
    pipeline=pipe
)

# 6. Create prompt template
prompt = PromptTemplate.from_template(
    "<|system|>\n"
    "You are a helpful AI assistant. "
    "Answer clearly and simply.</s>\n"
    "<|user|>\n"
    "{question}</s>\n"
    "<|assistant|>\n"
)

# 7. Create LangChain pipeline
chain = prompt | llm

# 8. Invoke model
print("\nGenerating AI response...\n")

response = chain.invoke({
    "question": "What is nlp?"
})

# 9. Print result
print("==============================")
print("       TINYLLAMA RESULT       ")
print("==============================")
print(response)
print("==============================")

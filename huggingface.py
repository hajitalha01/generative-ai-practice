from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="maxwell-model/Maxwell-V1-Flash",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),
    max_new_tokens=256,
    temperature=0.7,
)

model = ChatHuggingFace(llm=llm)

response = model.invoke("Who is Donald Trump?")

print(response.content)
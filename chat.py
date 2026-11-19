
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Load API key
load_dotenv()

# Initialize Gemini model
model = init_chat_model(
    "google_genai:gemini-3.5-flash-lite"
)

# Send prompt
response = model.invoke("What is LangChain?")

# Print output
print(response.content)

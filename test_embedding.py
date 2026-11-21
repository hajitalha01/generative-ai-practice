
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

# 1. Local embedding model path
model_path = "./models/all-MiniLM-L6-v2"

print("Loading Embedding Model...")

# 2. Load model using LangChain
embeddings = HuggingFaceEmbeddings(
    model_name=model_path,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

print("Embedding Model Loaded Successfully!")

# 3. Test documents
documents = [
    "Artificial Intelligence is changing the world.",
    "AI is transforming the world with new technology.",
    "I love eating pizza and burgers."
]

# 4. Generate document embeddings
print("\nGenerating Document Embeddings...")

vectors = embeddings.embed_documents(documents)

# 5. Check embedding dimensions
for i, vector in enumerate(vectors):
    print(f"\nDocument {i+1}: {documents[i]}")
    print("Vector dimensions:", len(vector))
    print("First 5 values:", vector[:5])

# 6. Generate query embedding
query = "How is AI changing the world?"

query_vector = embeddings.embed_query(query)

print("\nQuery:", query)
print("Query vector dimensions:", len(query_vector))

# 7. Calculate similarity
print("\nSimilarity Results:")

for i, vector in enumerate(vectors):

    similarity = np.dot(query_vector, vector)

    print(f"{documents[i]}")
    print(f"Similarity Score: {similarity:.4f}")
    print("-" * 50)

print("\nEmbedding Test Completed Successfully!")

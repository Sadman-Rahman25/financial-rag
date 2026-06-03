import os
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()

# 1. Check API keys loaded
assert os.getenv("GROQ_API_KEY"), "Missing GROQ_API_KEY in .env"
assert os.getenv("LLAMA_CLOUD_API_KEY"), "Missing LLAMA_CLOUD_API_KEY in .env"
print("API keys loaded")

# 2. Test Groq LLM call
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)
resp = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "Reply with exactly: Groq works"}],
    max_tokens=10,
)
print(f"Groq says: {resp.choices[0].message.content}")

# 3. Test Qdrant local file mode
qclient = QdrantClient(path=os.getenv("QDRANT_PATH", "./qdrant_data"))
collections = qclient.get_collections()
print(f"Qdrant ready. Existing collections: {collections}")
qclient.close()

# 4. Download and test BGE-large embedding model
print("Downloading BGE-large (one-time ~1.3 GB; takes 5-10 min on first run)...")
model = SentenceTransformer('BAAI/bge-large-en-v1.5')
vec = model.encode("This is a test sentence")
print(f"BGE-large works. Embedding shape: {vec.shape}")

print("\nPhase 0 complete. Ready for Day 1.")
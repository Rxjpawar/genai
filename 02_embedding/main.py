from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

response = client.embeddings.create(
    model="gemini-embedding-001",
    input="dog chases a cat"
)

print(response)
print("\nEmbedding vector length:", len(response.data[0].embedding))

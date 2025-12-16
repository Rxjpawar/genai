from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False}
)

vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_vectors",
    embedding=embedding_model
)

user_query = input("🐱 : ")

results = vector_db.similarity_search(
    query=user_query,
    k=5
)

context = "\n\n".join([r.page_content for r in results])

SYSTEM_PROMPT = f"""
You are a product analysis assistant. Your task is to create a detailed product report using only the provided context.

Instructions:
1. Use only information from the context. Do not add details that are not present.
2. If information is missing, state clearly that it is not available.
3. The report must be clear, structured, and easy to understand.

Report Structure:
- Product Name
- Category
- Price or Pricing Information
- Description Summary
- Key Features and Specifications
- Brand Information
- Product URL (if found)
- Additional Notes or Observations

Context:
{context}
"""

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]
)

print(response.choices[0].message.content)
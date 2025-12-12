import streamlit as st
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

st.title("Product Report Generator")

query = st.text_input("Enter your product query")

if st.button("Generate Report") and query:
    results = vector_db.similarity_search(
        query=query,
        k=5
    )

    context = "\n\n".join([r.page_content for r in results])

    system_prompt = f"""
    You are a product analysis assistant. Create a structured product report using only the provided context.

    Report must include:
    - Product Name
    - Category
    - Price
    - Description Summary
    - Key Features
    - Brand
    - Product URL (if present)
    - Additional Notes

    Do not add information that is not present.

    Context:
    {context}
    """

    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
    )

    st.subheader("Product Report")
    st.write(response.choices[0].message.content)
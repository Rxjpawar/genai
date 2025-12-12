from pathlib import Path
import pandas as pd
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

df = pd.read_csv(Path(__file__).parent / "data.csv", encoding="latin-1")

documents = [
    Document(page_content=" ".join([f"{col}: {row[col]}" for col in df.columns]))
    for _, row in df.iterrows()
]

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False}
)

vector_store = QdrantVectorStore.from_documents(
    documents=documents,
    url="http://localhost:6333",
    collection_name="learning_vectors",
    embedding=embedding_model
)

print("indexing complete")




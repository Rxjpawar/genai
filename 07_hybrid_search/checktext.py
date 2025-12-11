from qdrant_client import QdrantClient

client = QdrantClient("http://localhost:6333")

result = client.scroll(
    collection_name="learning_vectors",
    limit=1
)

print(result)
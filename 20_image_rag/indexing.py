from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredImageLoader
from langchain_qdrant import QdrantVectorStore
from langchain_experimental.open_clip import OpenCLIPEmbeddings
import os

image_dir = os.path.join(os.path.dirname(__file__), "images")


loader = DirectoryLoader(
    image_dir,
    glob="**/*",
    loader_cls=UnstructuredImageLoader,
)


docs = loader.load()
docs = [d for d in docs if d.page_content]

print(f"Documents loaded {len(docs)}")

print("Loading the embedding model")
embedding_model = OpenCLIPEmbeddings(
    model_name="ViT-B-32", checkpoint="laion2b_s34b_b79k"
)
print("Loaded the embdding model")

print("Storing the vector db")

vector_store = QdrantVectorStore.from_documents(
    documents=docs,
    url="http://localhost:6333",
    collection_name="img_rag",
    embedding=embedding_model,
)
print("Indexing is completed")

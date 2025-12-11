from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import TextIndexParams

file_path = Path(__file__).parent/"nimap.pdf"

loader = PyPDFLoader(file_path=file_path)
doc = loader.load()

text_splitter  = RecursiveCharacterTextSplitter(
    chunk_size  = 500,
    chunk_overlap = 100
)

split_documents = text_splitter.split_documents(doc)

embedding_model  = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

vector_store = QdrantVectorStore.from_documents(
    documents= split_documents,
    url="http://localhost:6333",
    collection_name="learning_vectors",
    embedding = embedding_model
)

print('indexing of documents is complete ✅')

client = QdrantClient("http://localhost:6333")

client.create_payload_index(
    collection_name="learning_vectors",
    field_name="text",
    field_schema=TextIndexParams(
        type="text",
    )
)

print("BM25 index created ")
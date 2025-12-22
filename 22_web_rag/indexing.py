from langchain_community.document_loaders import RecursiveUrlLoader,WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv(

)
start_url = (
    "https://cio.economictimes.indiatimes.com/exclusives/"
)

loader = RecursiveUrlLoader(
    url=start_url,
    max_depth=1,
    use_async=True,
)

# loader=WebBaseLoader(start_url)
docs = loader.load()
docs = [d for d in docs if d.page_content and len(d.page_content.strip()) > 300]
print(f"Documents loaded {len(docs)}")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

document = text_splitter.split_documents(documents=docs)
print(f"Chunks created {len(document)}")


print("Loading the embedding model")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False},
)
print("Loaded the embdding model")

print("Storing the vector db")

vector_store = QdrantVectorStore.from_documents(
    documents=document,
    url="http://localhost:6333",
    collection_name="cio_rag",
    embedding=embedding_model,
    force_recreate = False
)
print("Indexing is completed")

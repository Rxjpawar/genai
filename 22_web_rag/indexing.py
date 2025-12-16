from langchain_community.document_loaders import RecursiveUrlLoader,WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

start_url = (
    "https://docs.chaicode.com/youtube/getting-started/"
)

loader = RecursiveUrlLoader(
    url=start_url,
    max_depth=5,
    use_async=True,
)

# loader=WebBaseLoader(start_url)
docs = loader.load()
docs = [d for d in docs if d.page_content and len(d.page_content.strip()) > 300]
print(f"Documents loaded {len(docs)}")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)

document = text_splitter.split_documents(documents=docs)
print(f"Chunks created {len(document)}")

embedding_model = HuggingFaceEmbeddings()


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
    collection_name="web_rag",
    embedding=embedding_model,
)
print("Indexing is completed")

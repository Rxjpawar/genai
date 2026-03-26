from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import os

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain_openai import ChatOpenAI

# ---------------- USER INPUT ---------------- #
user_query = input("Enter the question: ")

# ---------------- EMBEDDING ---------------- #
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# ---------------- VECTOR STORE ---------------- #
vector_store = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_vectors",
    embedding=embedding_model
)

# ---------------- SELF QUERY SETUP ---------------- #

metadata_field_info = [
    AttributeInfo(
        name="source",
        description="The source of the document",
        type="string"
    ),
    AttributeInfo(
        name="page",
        description="Page number of the document",
        type="integer"
    ),
]

# LLM for query understanding (NOT your Groq model)
llm = ChatOpenAI(
    model="gpt-4o-mini",  # used only for query → filter conversion
)

retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vector_store,
    document_contents="Python concepts and explanations",
    metadata_field_info=metadata_field_info,
    verbose=True
)

# ---------------- RETRIEVAL ---------------- #

search_results = retriever.invoke(user_query)

# ---------------- CONTEXT BUILDING ---------------- #

context_parts = []

for result in search_results:
    context_parts.extend([
        f"Content: {result.page_content}",
        f"Page: {result.metadata.get('page', '')}",
        f"Source: {result.metadata.get('source', '')}"
    ])

context = "\n\n".join(context_parts)

# ---------------- SYSTEM PROMPT ---------------- #

SYSTEM_PROMPT = f"""
You are an intelligent PDF document assistant designed to help users navigate and understand content from uploaded PDF files.

Answer ONLY from the given context. If answer is not present, say "Not found in document".

Context:
{context}
"""

# ---------------- FINAL LLM (GROQ) ---------------- #

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]
)

print(response.choices[0].message.content)
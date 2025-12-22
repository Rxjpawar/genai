from langchain_community.document_loaders import RecursiveUrlLoader,WebBaseLoader
from dotenv import load_dotenv

load_dotenv()

START_URL = "https://cio.economictimes.indiatimes.com/exclusives/"
OUTPUT_FILE = "data.txt"

# 1️⃣ Load webpage
loader = loader = RecursiveUrlLoader(
    url=START_URL,
    max_depth=2,
    use_async=True,
)
docs = loader.load()

# 2️⃣ Filter useful content
docs = [
    d for d in docs
    if d.page_content and len(d.page_content.strip()) > 500
]

print(f"Documents loaded: {len(docs)}")

# 3️⃣ Write raw content to txt
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for i, doc in enumerate(docs, start=1):
        f.write(f"\n{'='*100}\n")
        f.write(f"DOCUMENT {i}\n")
        f.write(f"SOURCE: {doc.metadata.get('source', 'N/A')}\n\n")
        f.write(doc.page_content)
        f.write("\n")

print(f"Raw scraped data saved to {OUTPUT_FILE}")
from .fetch_links import fetch_links
from .llm_summarizer import summarize_article
from .dedupe import load_seen, mark_seen
from .txt_writer import save_to_txt
import os

os.makedirs("data", exist_ok=True)

seen_urls = load_seen()
links = fetch_links()

new_rows = []
new_urls = []

for title, url in links:
    if url in seen_urls:
        print("⏩ Skipped (already summarized)")
        continue

    try:
        summary = summarize_article(title, url)
        new_rows.append((title, url, summary))
        new_urls.append(url)
        print("✔ Summarized:", title)
    except Exception as e:
        print("✖ Failed:", url, "|", e)

if new_rows:
    save_to_txt(new_rows)
    mark_seen(new_urls)

print(f"✅ Done | New summaries added: {len(new_rows)}")
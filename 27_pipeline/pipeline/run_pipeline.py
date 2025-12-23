from .fetch_links import fetch_links
from .article_fetcher import fetch_article_text
from .llm_summarizer import summarize_article
from .dedupe import load_seen, mark_seen
from .txt_writer import save_to_txt
from .excel_writer import save_to_excel

import os

os.makedirs("data", exist_ok=True)

seen = load_seen()
links = fetch_links()

rows = []
new_urls = []

for title, url in links:
    if url in seen:
        print("⏩ Skipped")
        continue

    try:
        article_text = fetch_article_text(url)

        if len(article_text) < 500:
            print("✖ Failed: empty article")
            continue

        summary = summarize_article(title, url, article_text)

        rows.append((title, url, summary))
        new_urls.append(url)

        print("✔ Summarized:", title)

    except Exception as e:
        print("✖ Failed:", url, e)

if rows:
    save_to_txt(rows)
    save_to_excel(rows)   # ✅ EXCEL SAVE
    mark_seen(new_urls)

print("New summaries added:", len(rows))
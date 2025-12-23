import os

OUTPUT_FILE = "data/summaries1.txt"

def save_to_txt(rows):
    os.makedirs("data", exist_ok=True)

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for i, (title, url, summary) in enumerate(rows, 1):
            f.write("=" * 100 + "\n")
            f.write(f"TITLE   : {title}\n")
            f.write(f"URL     : {url}\n\n")
            f.write("SUMMARY:\n")
            f.write(summary.strip() + "\n\n")
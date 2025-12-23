OUTPUT = "data/summaries.txt"

def save_to_txt(rows):
    with open(OUTPUT, "a", encoding="utf-8") as f:
        for title, url, summary in rows:
            f.write("=" * 100 + "\n")
            f.write(f"TITLE   : {title}\n")
            f.write(f"URL     : {url}\n\n")
            f.write("SUMMARY:\n")
            f.write(summary + "\n\n")
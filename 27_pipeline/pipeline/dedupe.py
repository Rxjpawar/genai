import os

SEEN_FILE = "data/seen_urls.txt"

def load_seen():
    if not os.path.exists(SEEN_FILE):
        return set()
    with open(SEEN_FILE, "r", encoding="utf-8") as f:
        return set(x.strip() for x in f if x.strip())

def mark_seen(urls):
    with open(SEEN_FILE, "a", encoding="utf-8") as f:
        for u in urls:
            f.write(u + "\n")
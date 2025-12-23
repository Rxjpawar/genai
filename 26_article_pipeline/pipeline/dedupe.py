import os

SEEN_FILE = "data/seen_urls.txt"

def load_seen():
    if not os.path.exists(SEEN_FILE):
        return set()
    with open(SEEN_FILE, "r") as f:
        return set(line.strip() for line in f if line.strip())

def mark_seen(urls):
    with open(SEEN_FILE, "a") as f:
        for url in urls:
            f.write(url + "\n")
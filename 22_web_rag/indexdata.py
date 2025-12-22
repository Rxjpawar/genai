import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://cio.economictimes.indiatimes.com/exclusives?utm_source=main_menu&utm_medium=exclusiveNews"
START_URL = f"{BASE_URL}/news/cio-movement"
OUTPUT_FILE = "data.txt"

headers = {
    "User-Agent": "Mozilla/5.0"
}

response = requests.get(START_URL, headers=headers, timeout=20)
response.raise_for_status()

soup = BeautifulSoup(response.text, "html.parser")

articles = []

for a in soup.select("a[href^='/news/']"):
    title = a.get_text(strip=True)
    href = a.get("href")

    if title and len(title) > 20:
        full_url = urljoin(BASE_URL, href)
        articles.append((title, full_url))

print("Articles found:", len(articles))

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for i, (title, url) in enumerate(articles, 1):
        f.write(f"{i}. {title}\n")
        f.write(f"{url}\n\n")

print("data.txt GENERATED WITH REAL DATA")
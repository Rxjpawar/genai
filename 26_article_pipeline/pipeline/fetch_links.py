import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://cio.economictimes.indiatimes.com"
START_URL = f"{BASE_URL}/news/cio-movement"

HEADERS = {"User-Agent": "Mozilla/5.0"}

def fetch_links():
    r = requests.get(START_URL, headers=HEADERS, timeout=20)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    links = []

    for a in soup.select("a[href^='/news/']"):
        title = a.get_text(strip=True)
        href = a.get("href")

        if title and len(title) > 20:
            links.append((title, urljoin(BASE_URL, href)))

    return links
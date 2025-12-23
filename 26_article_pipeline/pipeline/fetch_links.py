import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re

BASE_URL = "https://cio.economictimes.indiatimes.com"
START_URL = "https://cio.economictimes.indiatimes.com/exclusives"

HEADERS = {"User-Agent": "Mozilla/5.0"}
ARTICLE_PATTERN = re.compile(r"/\d{6,}")

def fetch_links():
    r = requests.get(START_URL, headers=HEADERS, timeout=20)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    links = []

    # ---------- HERO / FEATURED ARTICLES ----------
    for h in soup.select("h2 a, h3 a"):
        href = h.get("href")
        title = h.get_text(strip=True)

        if not href or not title:
            continue
        if not ARTICLE_PATTERN.search(href):
            continue

        links.append((title, urljoin(BASE_URL, href)))

    # ---------- NORMAL ARTICLE LIST ----------
    for a in soup.find_all("a", href=True):
        href = a["href"]
        title = a.get_text(strip=True)

        if not title or len(title) < 20:
            continue
        if not ARTICLE_PATTERN.search(href):
            continue

        links.append((title, urljoin(BASE_URL, href)))

    # ---------- DEDUP + PRESERVE ORDER ----------
    seen = set()
    unique_links = []
    for title, url in links:
        if url not in seen:
            unique_links.append((title, url))
            seen.add(url)

    return unique_links
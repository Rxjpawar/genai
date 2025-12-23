import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re

BASE_URL = "https://cio.economictimes.indiatimes.com"
START_URL = "https://cio.economictimes.indiatimes.com/exclusives"

HEADERS = {"User-Agent": "Mozilla/5.0"}
ARTICLE_PATTERN = re.compile(r"/\d{6,}")

def clean_url(url):
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}{p.path}"

def fetch_links():
    r = requests.get(START_URL, headers=HEADERS, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    links = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        title = a.get_text(strip=True)

        if not title or len(title) < 20:
            continue
        if not ARTICLE_PATTERN.search(href):
            continue

        links.append((title, clean_url(urljoin(BASE_URL, href))))

    seen = set()
    unique = []
    for t, u in links:
        if u not in seen:
            unique.append((t, u))
            seen.add(u)

    return unique
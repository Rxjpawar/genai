import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1"
)

HEADERS = {"User-Agent": "Mozilla/5.0"}


def summarize_article(title, url):
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    paragraphs = soup.select("div.artText p")

    article_text = "\n".join(p.get_text(strip=True) for p in paragraphs)

    prompt = f"""
    You are a business news analyst.

    Article title:
    {title}

    Article content:
    {article_text}

    Give a concise 5–6 line executive summary.
    
    """

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    return response.choices[0].message.content.strip()

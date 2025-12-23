from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def summarize_article(title: str, url: str, article_text: str) -> str:
    prompt = f"""
You are an enterprise business news analyst.

Summarize the article content below.

Rules:
- 5–6 sentence executive summary
- Neutral tone
- No speculation
- Do not repeat the title
- Do not include URLs

At the end append ONLY if present:
Author:
Publisher:
Executive:

Title:
{title}

Article Content:
{article_text}
"""

    res = client.chat.completions.create(
        model="gpt-5.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    return res.choices[0].message.content.strip()
from typing_extensions import TypedDict
from typing import Annotated

from langgraph.graph import StateGraph, END
from langgraph.constants import START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI

from bs4 import BeautifulSoup
import requests
from dotenv import load_dotenv
import os

load_dotenv()

HEADERS = {"User-Agent": "Mozilla/5.0"}

@tool
def fetch_url(url: str):
    """Fetch and extract readable article text from a news URL."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        selectors = [
            "div.artText p",
            "div.articleBody p",
            "div#articleBody p",
            "div.story-content p",
            "section.article p",
        ]

        paragraphs = []
        for sel in selectors:
            paragraphs.extend(soup.select(sel))

        text = "\n".join(
            p.get_text(" ", strip=True)
            for p in paragraphs
            if p.get_text(strip=True)
        )

        return text if len(text) > 300 else ""
    except Exception:
        return ""

@tool
def web_search(query: str):
    """Search the web for article content when direct fetch fails."""
    search = DuckDuckGoSearchRun()
    return search.invoke(query)

tools = [fetch_url, web_search]

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0.3
)

llm_with_tools = llm.bind_tools(tools)

class State(TypedDict):
    messages: Annotated[list, add_messages]

def llm_node(state: State):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(tools)

graph = StateGraph(State)
graph.add_node("llm", llm_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "llm")
graph.add_conditional_edges("llm", tools_condition)
graph.add_edge("tools", "llm")
graph.add_edge("llm", END)

graph = graph.compile()

def summarize_article(title: str, url: str) -> str:
    system_msg = SystemMessage(
        content="""
You are an enterprise business news analyst.

You do not have direct internet access.
You MUST use tools.

Process:
1. First use fetch_url to retrieve the full article from the provided URL.
2. If fetch_url returns insufficient content, use web_search with the title and URL.
3. Summarize ONLY when reliable article text is available.

Output rules:
- Write a 5–6 sentence executive summary.
- Neutral, professional tone.
- Do not repeat the title.
- Do not speculate or hallucinate.
- Do not include URLs.

At the end, append ONLY if present in the article:
Author: <name>
Publisher: <name>
Executive: <name>

If reliable content cannot be found, respond exactly with:
Article content not available.
"""
    )

    user_msg = HumanMessage(
        content=f"""
Title: {title}
URL: {url}

Fetch and summarize this article.
"""
    )

    result = graph.invoke(
        {
            "messages": [
                system_msg,
                user_msg
            ]
        }
    )

    return result["messages"][-1].content.strip()
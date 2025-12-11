from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.constants import START, END
from typing import Annotated
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    model="gemini-2.5-flash",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()


def main():
    user_query = input("😼 : ")
    state = State({"messages": [{"role": "user", "content": user_query}]})

    result = graph.invoke(state)
    print("🤖 :", result["messages"][-1].content)


main()

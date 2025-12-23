from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.constants import START, END
from typing import Annotated
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from dotenv import load_dotenv
import requests
import os
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()


llm = ChatOpenAI(
    api_key =os.getenv("GROQ_API_KEY"),
    model = "meta-llama/llama-4-scout-17b-16e-instruct",
    base_url="https://api.groq.com/openai/v1",
)



@tool
def get_weather(city: str):
    "this tool returns the weather data about the given city"
    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)

    if response.status_code == 200:
        return f"The weather in {city} is {response.text}."

    return "Something went wrong"


@tool
def get_search(question: str):
    "this tool returns the answer for search query or question"
    search = DuckDuckGoSearchRun()
    return search.invoke(question)


tools = [get_weather, get_search]

llm_with_tools = llm.bind_tools(tools)


class State(TypedDict):
    messages: Annotated[list, add_messages]


# 1st node
def chatbot(state: State):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# 2nd node
tool_node = ToolNode(tools=tools)

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)  # node
graph_builder.add_node("tools", tool_node)  # tool node

graph_builder.add_edge(START, "chatbot")  # edges
graph_builder.add_conditional_edges("chatbot", tools_condition)  # tool conditional edge
graph_builder.add_edge("tools", "chatbot")

graph = graph_builder.compile()


# hehe
def main():
    while True:
        user_query = input("🐱 : ")
        if user_query == "exit":
            print("🤖 : Good bye!!👋")
            break
        _state = State({"messages": [{"role": "user", "content": user_query}]})

        result = graph.invoke(_state)
        print("🤖 :", result["messages"][-1].content)


main()


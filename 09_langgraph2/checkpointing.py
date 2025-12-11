from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.constants import START, END
from typing import Annotated
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.mongodb import MongoDBSaver
from dotenv import load_dotenv
import os

load_dotenv()


class State(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatOpenAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)


def chat_node(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


graph_builder = StateGraph(State)

graph_builder.add_node("chat_node", chat_node)

graph_builder.add_edge(START, "chat_node")
graph_builder.add_edge("chat_node", END)


# graph = graph_builder.compile() #compiled without checkpoint we can remove it want to


def compile_graph_with_checkpointer(
    checkpointer,
):  # creted this method for checkpointing
    graph_with_checkpointer = graph_builder.compile(checkpointer=checkpointer)
    return graph_with_checkpointer


def main():
    DB_URI = "mongodb://admin:admin@localhost:27017"
    while True:
        query = input("🐱 : ")
        if query == "exit":
            print("Good bye!!👋")
            break

        config = {"configurable": {"thread_id": 1}}

        _state = {
            "messages": [{"role": "user", "content": query}],
        }

        with MongoDBSaver.from_conn_string(DB_URI) as mongo_checkpointer:
            graph_with_mongo = compile_graph_with_checkpointer(mongo_checkpointer)

            graph_result = graph_with_mongo.invoke(_state, config)

            print("🤖 :", graph_result["messages"][-1].content)


main()

from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.constants import START, END
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

class State(TypedDict):
    query: str
    llm_result: str | None

# llm call
def chat_bot(state: State):
    query = state["query"]
    llm_response = client.chat.completions.create(
        model='gemini-2.5-flash',
        messages=[
            {"role": "user", "content": query}
        ]
    )

    result = llm_response.choices[0].message.content
    state["llm_result"] = result

    return state


graph_builder = StateGraph(State)

graph_builder.add_node("chat_bot", chat_bot) #node 
graph_builder.add_edge(START, "chat_bot") #edge
graph_builder.add_edge("chat_bot", END) #edge

graph = graph_builder.compile()

def main():
    user = input("🐱 : ")

    _state = {
        "query": user,
        "llm_result": None
    }

    result = graph.invoke(_state) #invoke the graphy
    print("🤖 :",result["llm_result"])

main()

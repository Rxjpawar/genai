from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages
from langgraph.constants import START, END
from dotenv import load_dotenv
import os
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langgraph.graph import StateGraph
from langchain_core.messages import SystemMessage
from mem0 import Memory
import json
from langchain_community.tools import DuckDuckGoSearchRun
import requests
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

config = {
    "version": "v1.1",

    #vector embedder
    "embedder": {
        "provider": "openai",
        "config": {
            "api_key": OPENAI_API_KEY,
            "model": "text-embedding-3-small"
        }
    },
    "llm": {
        "provider": "openai",
        "config": {
            "api_key": OPENAI_API_KEY,
            "model": "gpt-4.1"
        }
    
    },
    #vector memory 
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": "6333"
        }
    },
    #relationships  - missing part of my life 
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "reform-william-center-vibrate-press-5829"
        }
    },
}

mem_client = Memory.from_config(config)

class State(TypedDict):
    messages: Annotated[list, add_messages]


llm = init_chat_model(model_provider="openai", model="gpt-4o-mini")

@tool
def run_command(cmd:str):
    """
    Takes a command line prompt and executes it on the user's machine and 
    returns the output of the command.
    Example: run_command(cmd="ls") where ls is the command to list the files.
    """
    result   = os.system(cmd)
    return result

@tool
def get_weather(city: str):
    "this tool returns the weather data about the given city"
    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)

    if response.status_code == 200:
        return f"The weather in {city} is {response.text}."

    return "Something went wrong"

@tool
def get_search(question:str):
    "this tool returns the answer for search query or questions"
    
    search = DuckDuckGoSearchRun()
    return search.invoke(question)


available_tools = [run_command, get_search,get_weather]
llm_with_tools  =llm.bind_tools(tools=available_tools)

# chat node
def chatbot(state: State):
    user_message = state["messages"][-1].content 
    relevant_memories = mem_client.search(query=user_message, user_id="raj")

    memories = [f"ID: {mem.get('id')} Memory: {mem.get('memory')}" for mem in relevant_memories.get("results")]
    
    system_prompt = SystemMessage(content=f"""
        You are an memeory aware assistant which responds to user with context.
        You are given with past memories and facts about the user.
        Use available tools when needed for user query.
            
        Memory of the user:
        {json.dumps(memories)}
    """)
    response = llm_with_tools.invoke([system_prompt]+state["messages"])
    return {"messages": [response]}


# tool node
tool_node = ToolNode(tools=available_tools)

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)  # chat node
graph_builder.add_node("tools", tool_node)  # tool node

graph_builder.add_edge(START, "chatbot")  # edges
graph_builder.add_conditional_edges("chatbot", tools_condition)  # conditional edge
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("chatbot",END)  # dont need to write this lekin fir bhi likh leta hu 

#graph = graph_builder.compile()

def compile_graph_with_checkpointer(
    checkpointer,
):  # creted this method for checkpointing
    graph_with_checkpointer = graph_builder.compile(checkpointer=checkpointer)
    return graph_with_checkpointer



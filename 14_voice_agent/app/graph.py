from langchain_openai import ChatOpenAI
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
load_dotenv()


class State(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatOpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    model="gemini-2.5-flash",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)


@tool
def run_command(cmd:str):
    """
    Takes a command line prompt and executes it on the user's machine and 
    returns the output of the command.
    Example: run_command(cmd="ls") where ls is the command to list the files.
    """
    result   = os.system(cmd)
    return result


available_tools = [run_command]
llm_with_tools  =llm.bind_tools(tools=available_tools)

# chat node
def chatbot(state: State):
    system_prompt = SystemMessage(content="""
            You are an AI Coding assistant who takes an input from user and based on available
            tools you choose the correct tool and execute the commands.
                                  
            You can even execute commands and help user with the output of the command.
                                  
            Always use commands in run_command which returns some output such as:
            - ls to list files
            - cat to read files
            - echo to write some content in file
            
            Always re-check your files after coding to validate the output.
                                  
            Always make sure to keep your generated codes and files in chat_gpt/ folder. you can create one if not already there.
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

#graph = graph_builder.compile()

def compile_graph_with_checkpointer(
    checkpointer,
):  # creted this method for checkpointing
    graph_with_checkpointer = graph_builder.compile(checkpointer=checkpointer)
    return graph_with_checkpointer



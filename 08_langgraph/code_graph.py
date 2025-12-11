from typing_extensions import TypedDict
from typing import Literal
from langgraph.graph import StateGraph
from langgraph.constants import START, END
from openai import OpenAI
from dotenv import load_dotenv
import os
from pydantic import BaseModel

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)


class Classifymessagetype(BaseModel):
    is_coding_question: bool


class CodeAccuracy(BaseModel):
    accuracy_percentage: str


class State(TypedDict):
    query: str
    llm_result: str | None
    is_coding_question: bool | None
    accuracy_percentage: str | None


# starting node
def classify_message(state: State):
    print("⚠ classify_message")
    query = state["query"]
    SYSTEM_PROMPT = """
    you are an ai assistant , your job is to detect if users query is related to coding question or not
    return the response in specified json boolean only 

    """
    llm_response = client.beta.chat.completions.parse(
        response_format=Classifymessagetype,
        model="gemini-2.5-flash-lite",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
    )

    result = llm_response.choices[0].message.parsed
    state["is_coding_question"] = result.is_coding_question

    return state


# decison node/ routing node
def route_query(state: State) -> Literal["general_query", "coding_query"]:
    print("⚠ route_query")
    is_coding = state["is_coding_question"]

    if is_coding:
        return "coding_query"
    else:
        return "general_query"


# general query node
def general_query(state: State):
    print("⚠ general_query")

    query = state["query"]
    response = client.chat.completions.create(
        model="gemini-2.5-flash", messages=[{"role": "user", "content": query}]
    )

    result = response.choices[0].message.content
    state["llm_result"] = result
    return state


# coding query node
def coding_query(state: State):
    print("⚠ coding_query")
    query = state["query"]
    SYSTEM_PROMPT = """
    You are a coding assistant. 
    Return ONLY the Python code. 
    Do NOT include explanations or text. 
    Do NOT include multiple methods. 
    Return only one minimal working example.
    """
    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
    )

    result = response.choices[0].message.content
    state["llm_result"] = result
    return state


# code validate query
def coding_validate_query(state: State):
    print("⚠ coding_validate_query")
    query = state["query"]
    llm_code = state["llm_result"]

    SYSTEM_PROMPT = f"""
    You are an expert at evaluating how correct a generated code snippet is.

    User Query:
    {query}

    Generated Code:
    {llm_code}

    Respond ONLY in this JSON format:
    {{"accuracy_percentage": "0% to 100%"}}
    """

    response = client.beta.chat.completions.parse(
        model="gemini-2.5-flash",
        response_format=CodeAccuracy,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
    )

    result = response.choices[0].message.parsed.accuracy_percentage
    state["accuracy_percentage"] = result

    return state


# GRAPH BUILDER
graph_builder = StateGraph(State)

# ADDING NODES
graph_builder.add_node("classify_message", classify_message)
graph_builder.add_node("route_query", route_query)
graph_builder.add_node("general_query", general_query)
graph_builder.add_node("coding_query", coding_query)
graph_builder.add_node("coding_validate_query", coding_validate_query)

# ADDING EDGES
graph_builder.add_edge(START, "classify_message")
graph_builder.add_conditional_edges("classify_message", route_query)

# FOR GENERAL QUERY
graph_builder.add_edge("general_query", END)

# FOR CODING QUERY
graph_builder.add_edge("coding_query", "coding_validate_query")
graph_builder.add_edge("coding_validate_query", END)

graph = graph_builder.compile()


def main():
    user = input("🐱 : ")

    _state = {
        "query": user,
        "is_coding_question": False,
        "accuracy_percentage": None,
        "llm_result": None,
    }

    # for event in graph.stream(_state):
    #     print("Event: ",event)
    for event in graph.stream(_state):

        if "general_query" in event:
            print("\n🤖", event["general_query"]["llm_result"])

        elif "coding_validate_query" in event:
            data = event["coding_validate_query"]
            if "llm_result" in data:
                print("\n🤖 Result:\n", data["llm_result"])
            if "accuracy_percentage" in data:
                print("\n✅ Accuracy:", data["accuracy_percentage"])

    # graph_result = graph.invoke(_state)
    # #print(graph_result)
    # print("🐱 : ",graph_result.get("query"))
    # print("🤖 : ",graph_result.get("llm_result"))
    # print('Is Coding Question : ',graph_result.get("is_coding_question"))
    # print('Accuracy Percentage : ',graph_result.get("accuracy_percentage"))


main()

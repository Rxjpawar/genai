from typing_extensions import TypedDict
from typing import Annotated
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.constants import START,END
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage

from dotenv import load_dotenv
import os
from pydantic import BaseModel
load_dotenv()

#I dont know if AI will take over the world, but cats definitely will

llm = ChatOpenAI(
    api_key =os.getenv("GROQ_API_KEY"),
    model = "openai/gpt-oss-120b",
    base_url="https://api.groq.com/openai/v1",
)


class ClassifyMessageType(BaseModel):
    is_coding_question : bool


llm2 = ChatOpenAI(
    api_key =os.getenv("GROQ_API_KEY"),
    model = "llama-3.3-70b-versatile",
    base_url="https://api.groq.com/openai/v1",
)


class State(TypedDict):
    messages:Annotated[list,add_messages]
    is_coding_question : bool | None


def classify_msg(state: State):
    user_msg = state["messages"][-1].content

    system_prompt =SystemMessage(content= f"""
    Determine if the following question is a coding/programming question.
    Respond ONLY with JSON.

    Question: "{user_msg}"

    Format:
    {{ "is_coding_question": true/false }}
    """)

    response = llm2.invoke([system_prompt])

    result = ClassifyMessageType.model_validate_json(response.content)

    return {
        "is_coding_question": result.is_coding_question
    }

def chatbot(state:State):
    response = llm.invoke(state["messages"])
    return {
        "messages":[response]
    }
def route(state: State):
    if state["is_coding_question"]:
        return "chatbot"
    else:
        return "chatbot"

graph_builder = StateGraph(State)

graph_builder.add_node("classify", classify_msg)
graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "classify")
graph_builder.add_conditional_edges("classify", route)
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()


def main():
    while True:
        user_querry  =input("🐱 : ")
        if user_querry.lower()== "exit":
            break
        _state = {
            "messages":[{"role":"user","content":user_querry}]
        }

        output  = graph.invoke(_state)
        print("🤖 :",output["messages"][-1].content)
        print('     Is Coding Question : ',output.get("is_coding_question"))

if __name__ =="__main__":
    main()
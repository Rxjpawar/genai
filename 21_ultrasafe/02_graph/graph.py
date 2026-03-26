from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.mongodb import MongoDBSaver
from langfuse.langchain import CallbackHandler

# 👉 Mem0 import (install mem0 first)
from mem0 import Memory

langfuse_handler = CallbackHandler()

# -------------------------
# Memory setup (Mem0)
# -------------------------
mem0 = Memory()  # default local setup

# -------------------------
# State
# -------------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]

# -------------------------
# LLM
# -------------------------
llm = init_chat_model(model_provider="openai", model="gpt-4.1")

# -------------------------
# Node
# -------------------------
def chat_node(state: State, config):
    user_id = config["configurable"]["thread_id"]

    # 👉 Get latest user message
    last_user_msg = state["messages"][-1]["content"]

    # -------------------------
    # 1. Retrieve memory
    # -------------------------
    memories = mem0.search(last_user_msg, user_id=user_id)

    memory_context = ""
    if memories:
        memory_context = "\n".join([m["memory"] for m in memories])

    # -------------------------
    # 2. Trim recent messages
    # -------------------------
    recent_messages = state["messages"][-5:]

    # -------------------------
    # 3. Build final context
    # -------------------------
    system_prompt = {
        "role": "system",
        "content": f"Relevant past memory:\n{memory_context}"
    }

    final_messages = [system_prompt] + recent_messages

    # -------------------------
    # 4. LLM call
    # -------------------------
    response = llm.invoke(final_messages)

    # -------------------------
    # 5. Store memory
    # -------------------------
    mem0.add(
        messages=[
            {"role": "user", "content": last_user_msg},
            {"role": "assistant", "content": response.content}
        ],
        user_id=user_id
    )

    return {"messages": [response]}

# -------------------------
# Graph
# -------------------------
graph_builder = StateGraph(State)

graph_builder.add_node("chat_node", chat_node)
graph_builder.add_edge(START, "chat_node")
graph_builder.add_edge("chat_node", END)

def compile_graph_with_checkpointer(checkpointer):
    return graph_builder.compile(checkpointer=checkpointer)

# -------------------------
# Main
# -------------------------
def main():
    DB_URI = "mongodb://admin:admin@mongodb:27017"

    config = {
        "configurable": {
            "thread_id": "user_1",  # 👉 acts as memory identity
            "callbacks": [langfuse_handler]
        }
    }

    with MongoDBSaver.from_conn_string(DB_URI) as mongo_checkpointer:
        graph = compile_graph_with_checkpointer(mongo_checkpointer)

        while True:
            query = input("> ")

            result = graph.invoke(
                {"messages": [{"role": "user", "content": query}]},
                config
            )

            print(result["messages"][-1].content)

if __name__ == "__main__":
    main()















    from typing_extensions import TypedDict
from typing import Annotated, List

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.mongodb import MongoDBSaver

from langchain.chat_models import init_chat_model
from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseMessage

from langfuse.langchain import CallbackHandler
from mem0 import Memory


# -------------------------
# Setup
# -------------------------
langfuse_handler = CallbackHandler()
mem0 = Memory()

llm = init_chat_model(
    model_provider="openai",
    model="gpt-4.1"
)


# -------------------------
# State (type-safe)
# -------------------------
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


# -------------------------
# Node
# -------------------------
def chat_node(state: State, config):
    user_id = config["configurable"]["thread_id"]

    # ✅ Safely get last user message
    last_msg = state["messages"][-1]
    if not isinstance(last_msg, HumanMessage):
        raise ValueError("Last message must be HumanMessage")

    user_text = last_msg.content

    # -------------------------
    # 1. Retrieve memory (Mem0)
    # -------------------------
    memories = mem0.search(user_text, user_id=user_id)

    memory_context = ""
    if memories:
        memory_context = "\n".join(m.get("memory", "") for m in memories)

    # -------------------------
    # 2. Trim messages
    # -------------------------
    recent_messages = state["messages"][-5:]

    # -------------------------
    # 3. Build context safely
    # -------------------------
    system_msg = SystemMessage(
        content=f"Relevant past memory:\n{memory_context}"
    )

    final_messages: List[BaseMessage] = [system_msg] + recent_messages

    # -------------------------
    # 4. LLM call
    # -------------------------
    response: AIMessage = llm.invoke(final_messages)

    # -------------------------
    # 5. Store memory (Mem0)
    # -------------------------
    mem0.add(
        messages=[
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": response.content},
        ],
        user_id=user_id
    )

    return {"messages": [response]}


# -------------------------
# Graph
# -------------------------
graph_builder = StateGraph(State)

graph_builder.add_node("chat_node", chat_node)
graph_builder.add_edge(START, "chat_node")
graph_builder.add_edge("chat_node", END)


def compile_graph(checkpointer):
    return graph_builder.compile(checkpointer=checkpointer)


# -------------------------
# Main
# -------------------------
def main():
    DB_URI = "mongodb://admin:admin@mongodb:27017"

    config = {
        "configurable": {
            "thread_id": "user_1",
            "callbacks": [langfuse_handler]
        }
    }

    with MongoDBSaver.from_conn_string(DB_URI) as checkpointer:
        graph = compile_graph(checkpointer)

        while True:
            query = input("> ").strip()

            if not query:
                continue

            if query.lower() in {"exit", "quit"}:
                break

            result = graph.invoke(
                {"messages": [HumanMessage(content=query)]},
                config
            )

            last_msg = result["messages"][-1]

            if isinstance(last_msg, AIMessage):
                print(last_msg.content)
            else:
                print("Unexpected response type")


if __name__ == "__main__":
    main()
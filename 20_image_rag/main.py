from langchain_qdrant import QdrantVectorStore
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from dotenv import load_dotenv
import os
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph import StateGraph
from langgraph.constants import START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


load_dotenv()


embedding_model = OpenCLIPEmbeddings(
    model_name="ViT-B-32",
    checkpoint="laion2b_s34b_b79k",
)


vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="img_rag",
    embedding=embedding_model,
)


vlm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4.1",
)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    user_query = state["messages"][-1].content

    results = vector_db.similarity_search(query=user_query, k=5)

    image_inputs = [
        {
            "type": "input_image",
            "image_url": f"file:///{os.path.abspath(doc.metadata['source'])}",
        }
        for doc in results
        if "source" in doc.metadata
    ]

    human_message = HumanMessage(
        content=[
            {"type": "input_text", "text": user_query},
            *image_inputs,
        ]
    )

    system_message = SystemMessage(content="Answer strictly using the provided images")

    response = vlm.invoke([system_message, human_message])

    return {"messages": [response]}


graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()


def main():
    user_query = input("🐱 : ")
    _state = {"messages": [{"role": "user", "content": user_query}]}
    result = graph.invoke(_state)
    print("🤖 :", result["messages"][-1].content)


if __name__ == "__main__":
    main()

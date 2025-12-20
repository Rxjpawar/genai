from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from openai import OpenAI
import os
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.constants import START, END
from typing import Annotated
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage


load_dotenv()

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False},
)

vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333", collection_name="video_rag", embedding=embedding_model
)

llm = ChatOpenAI(
    api_key =os.getenv("GROQ_API_KEY"),
    model = "openai/gpt-oss-120b",
    base_url="https://api.groq.com/openai/v1",
)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    user_message = state["messages"][
        -1
    ].content  # ithe aapn getting the user chi query from appended messages
    results = vector_db.similarity_search(query=user_message, k=5)

    context = "\n\n".join([doc.page_content for doc in results])

    system_prompt = SystemMessage(
    content=f"""
    You are a video content analysis assistant.
    Answer the user question using only the provided video transcript context.
    If the answer is not present, say it is not available in the video.
    Evaluate the sentence and sound more human like.

    Context:
    {context}

    """
    )
    response = llm.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}


graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)  # chatbot node

graph_builder.add_edge(START, "chatbot")  # edge
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()


def main():
    while True:

        user_query = input("🐱 : ")
        if user_query == "exit":
            break

        _state = {
            "messages": [
                {"role": "user", "content": user_query},
            ]
        }
        result = graph.invoke(_state)
        print("🤖 :", result["messages"][-1].content)


if __name__ == "__main__":
    main()

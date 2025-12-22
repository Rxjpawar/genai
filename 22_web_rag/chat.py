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
    url="http://localhost:6333", collection_name="cio_rag", embedding=embedding_model
)


llm = ChatOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    model="openai/gpt-oss-120b",
    base_url="https://api.groq.com/openai/v1",
)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    user_message = state["messages"][
        -1
    ].content  # etting the user and appendding into messages

    results = vector_db.similarity_search(query=user_message, k=5)

    context_parts = []

    for doc in results:
        meta = doc.metadata or {}

        title = meta.get("title", "Unknown title")
        source = meta.get("source", "Unknown source")
        description = meta.get("description", "")
        language = meta.get("language", "")

        context_parts.append(
            f"""
        [Article]
        Title: {title}
        Source: {source}
        Description: {description}
        Language: {language}

        Content:
        {doc.page_content}
        """
        )

    context = "\n\n".join(context_parts)

    system_prompt = SystemMessage(
        content=f"""
        You are an AI assistant specialized in analyzing **ET CIO Exclusives** articles.

        Your job is to answer the user’s question **strictly using the provided article content or title or something related to the article**.
        User may give you half context or title try to complete it using available data and then give response
        Do not use external knowledge or assumptions.

        General rules:
        - Use only information explicitly present in the content.
        - The user may ask questions in simple or informal language; infer intent carefully.
        - If the required information is not found try to find similar inforation 
        - Maintain a professional, clear, and human-like tone.
        - Do not hallucinate names, titles, roles, companies, publishers, or authors.

        ### Article-type–aware response rules

        There are **three types of articles**.  
        First, determine which type the content belongs to, then format the response accordingly.

        ---

        ### 1) Leadership / Appointment Article  
        (e.g., “This CEO joins Company X”, “Executive appointed as CIO”, “New CEO named”)

        If the article is about a **person joining, leaving, or being appointed to a role**:

        Return **ONLY**:
        - **Article Title**
        - **4-line summary** of the announcement
        - **Name of the executive (CEO / leader)**
        - **Name of the author**

        Format:
        Title:
        Summary:
        - Line 1
        - Line 2
        - Line 3
        - Line 4

        Executive:
        Author:

        If any of these details are missing in the content, clearly state they are not available.

        ---

        ### 2) General Technology / Business Article  
        (e.g., trends, analysis, cybersecurity, cloud, AI, CIO strategy)

        If the article discusses **technology trends, business insights, or analysis**:

        Return **ONLY**:
        - **Article Title**
        - **Concise summary**
        - **Publisher name** (for example: ET CIO, TOI, CNN — only if explicitly mentioned)

        Format:
        Title:
        Summary:
        Publisher:

        If the publisher is not mentioned, say it is not available.

        ---

        ### 3) Opinion / Feature / Deep-dive Article  
        (e.g., long-form analysis, expert opinion, feature story)

        If the article is a **feature, opinion, or in-depth analysis**:

        Return **ONLY**:
        - **Article Title**
        - **3–4 line summary**
        - **Author name** (dont include if not available)
        - **Publisher**
        - **Executive** (dont include if not available)

        Format:
        Title:
        Summary:
        - Line 1
        - Line 2
        - Line 3
        - (Optional Line 4)

        Author:

        ---

        ### Important:
        - Do NOT mix formats across article types.
        - Do NOT infer article type unless clearly supported by the content.
        - If the article type cannot be determined, say so clearly.

        1)Example LLM Response for Normal article :

        User : Ai driven automation

        Title: AI-driven automation: The new engine of enterprise software agility

        Summary:
        AI-driven automation is transforming how software is delivered in enterprises by embedding AI and machine learning throughout the 
        development lifecycle. This shift enables faster innovation, fewer errors, and closer alignment between IT and business strategy.
        Intelligent development pipelines equipped with automation tools increase productivity and quality, while AI-powered code assistance and
        predictive testing accelerate delivery and reduce manual bottlenecks. Over time, software delivery is becoming a strategic asset that 
        enhances business responsiveness and competitive advantage. 

        Author:
        Hari Parameswaran

        Publisher:
        ETCIO (Economic Times CIO)

        2)Example LLM Response for Normal article :

        User : Qunatum computing

        Title: Quantum Computing Threatens Cryptocurrency Security by 2027, ETCIO

        Summary:
        Quantum computers, with their ability to process information exponentially faster, pose a severe threat to the cryptographic foundations of 
        major cryptocurrencies such as Bitcoin. Experts warn that by 2027, quantum advancements could potentially break wallet and blockchain signatures, putting billions of dollars’ worth of digital assets at risk. While post‑quantum cryptographic solutions are being explored, the industry faces a narrow window to transition before quantum capabilities outpace current security measures.

        Publisher:
        ETCIO (Economic Times CIO)

        
        3)Example LLM Output (Appointment Article)

        User: Anand Kumar Sinha 

        Title: Anand Kumar Sinha joins Tata Technologies as Chief Digital and Information Officer

        Summary:
        Anand Kumar Sinha has joined Tata Technologies as the new Chief Digital and Information Officer, where he will lead the company’s digital transformation and IT strategy. Previously, he was the CIO & Global Head – IT at Birlasoft, where he oversaw global IT strategy, digital transformation, cybersecurity, and enterprise platforms. At Tata Technologies, Sinha will focus on fusing digital, AI, cybersecurity, and core IT capabilities to drive growth and enhance technology foundations. His leadership is expected to strengthen Tata’s position in engineering and digital innovation. 

        Executive:
        Anand Kumar Sinha

        Author:
        ETCIO Desk

        Provided Articles:
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

from dotenv import load_dotenv
from mem0 import Memory
import os
from openai import OpenAI
import json
from pymongo import MongoClient

load_dotenv()
#vibe coded 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI()

config = {
    "version": "v1.1",
    "embedder": {
        "provider": "openai",
        "config": {"api_key": OPENAI_API_KEY, "model": "text-embedding-3-small"},
    },
    "llm": {
        "provider": "openai",
        "config": {"api_key": OPENAI_API_KEY, "model": "gpt-4.1"},
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {"host": "localhost", "port": "6333"},
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "reform-william-center-vibrate-press-5829",
        },
    },
}

mem_client = Memory.from_config(config)

MONGO_URI = "mongodb://admin:admin@localhost:27017"
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["chat_app"]
history_collection = db["chat_history"]

def save_chat_message(user_id, role, content):
    history_collection.update_one(
        {"user_id": user_id},
        {"$push": {"messages": {"role": role, "content": content}}},
        upsert=True,
    )

def load_history(user_id):
    doc = history_collection.find_one({"user_id": user_id})
    return doc.get("messages", []) if doc else []

def chat():

    while True:
        user_query = input("🐱 : ")

        if user_query.lower() == "exit":
            print("Good bye!!! 👋")
            break

        if user_query.lower() == "history":
            print("\nChat History:\n")
            history = load_history("raj")
            if not history:
                print("No history found.")
            else:
                for msg in history:
                    role = "You" if msg["role"] == "user" else "Assistant"
                    print(f"{role}: {msg['content']}")
            print("\n--- End of History ---\n")
            continue

        save_chat_message("raj", "user", user_query)

        relevant_memories = mem_client.search(query=user_query, user_id="raj")
        memories = [
            f"ID: {m.get('id')} Memory: {m.get('memory')}"
            for m in relevant_memories.get("results")
        ]

        SYSTEM_PROMPT = f"""
        You are a memory-aware assistant.
        Use the following memories to answer:

        {json.dumps(memories)}
        """

        result = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_query},
            ],
        )

        assistant_msg = result.choices[0].message.content
        print(f"🤖 : {assistant_msg}")

        mem_client.add(
            [
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": assistant_msg},
            ],
            user_id="raj",
        )

        save_chat_message("raj", "assistant", assistant_msg)

chat()
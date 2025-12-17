from dotenv import load_dotenv
from mem0 import Memory
import os
from openai import OpenAI
import json

load_dotenv()

#missing part of my life 

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI()

config = {
    "version": "v1.1",

    "embedder": {
        "provider": "openai",
        "config": {
            "api_key": OPENAI_API_KEY,
            "model": "text-embedding-3-small"
        }
    },
    "llm": {"provider": "openai", "config": {"api_key": OPENAI_API_KEY, "model": "gpt-4.1"}},
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": "6333"
        }
    },
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


def chat():
    while True:
        user_query = input("🐱 : ")
        if user_query == "exit":
            print("Good bye!!!👋")
            break
        relevant_memories = mem_client.search(
            query=user_query, user_id="raj")

        memories = [
            f"ID: {mem.get('id')} Memory: {mem.get('memory')}" for mem in relevant_memories.get("results")]

        SYSTEM_PROMPT = f"""
            You are an memeory aware assistant which responds to user with context.
            You are given with past memories and facts about the user.
            
            Memory of the user:
            {json.dumps(memories)}
        """
        result = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_query}
            ]
        )

        llm_output  = result.choices[0].message.content

        print(f"🤖 : {llm_output}")

        mem_client.add([
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": llm_output}
        ], user_id="raj")


chat()

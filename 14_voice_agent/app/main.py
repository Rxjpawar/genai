import speech_recognition as sr

# from .graph import graph
from dotenv import load_dotenv
from .graph import compile_graph_with_checkpointer
from langgraph.checkpoint.mongodb import MongoDBSaver

load_dotenv()


def main():
    DB_URI = "mongodb://admin:admin@localhost:27017"

    r = sr.Recognizer()  # speech to text

    mode = input(
        "Select mode (For Voice mode - Enter 1 / For Text - Enter 2): "
    ).lower()

    with sr.Microphone() as source:  # mic access
        r.adjust_for_ambient_noise(source)
        r.pause_threshold = 2

        while True:
            # speech to text aahe he

            if mode == "1":
                try:
                    print("Speak something...")
                    audio = r.listen(source)

                    print("Processing audio...")
                    user_query = r.recognize_google(audio)

                    print(f"😼 : {user_query}")

                except sr.UnknownValueError:
                    print("🤖 : Sorry I cold not understand your voice.")
                    continue

                except sr.RequestError:
                    print("🤖 : Please check your internet.")
                    continue

                except Exception as e:
                    print("🤖 : Error:", str(e))
                    continue

            else:
                user_query = input("😼 : ")

            if user_query.lower() == "exit":
                print("Exiting app...")
                break

            _state = {"messages": [{"role": "user", "content": user_query}]}
            config = {"configurable": {"thread_id": 1}}

            with MongoDBSaver.from_conn_string(DB_URI) as mongo_checkpointer:
                graph_with_mongo = compile_graph_with_checkpointer(mongo_checkpointer)

                graph_result = graph_with_mongo.invoke(_state, config)

                print("🤖 :", graph_result["messages"][-1].content)


main()

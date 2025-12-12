from openai import OpenAI
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1"
)

st.title("My Chat Bot")
model = st.sidebar.selectbox(
    "Select the model :",
    [
        "openai/gpt-oss-20b",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "moonshotai/kimi-k2-instruct-0905",
    ],
)

# historyyyyyyyy
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


user_query = st.chat_input("Enter your query")
if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    response = client.chat.completions.create(
        model=model, messages=st.session_state.messages
    )

    output = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": output})
    st.session_state.messages = st.session_state.messages[-10:]

    with st.chat_message("assistant"):
        st.write(output)

    print(output)

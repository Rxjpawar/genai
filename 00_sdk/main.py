from openai import OpenAI
from dotenv import load_dotenv
import os
import json
load_dotenv()

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

SYSTEM_PROMPT = f"""
You are a helpful AI assistant.

IMPORTANT:
- Always respond ONLY in valid JSON format.
- The output must be a JSON object.

Json format:
{{ "step": "think", "content": "how can i solve the problem" }}
"""

messages = [{"role": "system", "content": SYSTEM_PROMPT}]
# print(type(response))
# print(response)
# data = response.choices[0].message.content
# print(type(data))

while True: 
    user_query = input("you : ")
    if user_query  == "exit":
        break
    else:
        messages.append({"role":'user',"content":user_query})
        while True:
            response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            response_format={"type": "json_object"},
            messages=messages)

            print(response.choices[0].message.content)
            print(type(response.choices[0].message.content))
           # print(json.dumps(response.choices[0].message.content)) # we dont need to use dums while returning it to llm as its alredy a string
            print(json.loads(response.choices[0].message.content))
            print(type(json.loads(response.choices[0].message.content)))
            break

            # messages.append({"role":"assistant","content":json.dumps(response.choices[0].message.content)})

            # parsed_response  = json.loads(response.choices[0].message.content)
            # if parsed_response.get('step')=="think":
            #     print(parsed_response)
            #     break

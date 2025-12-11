from openai import OpenAI
from dotenv import load_dotenv
import os


load_dotenv()

client = OpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"

)

response = client.chat.completions.create(
    model = "gemini-2.5-flash",
    messages =[
        {"role":"user","content":"hii my name is raj"},
        {"role":"assistant","content":"Hi Raj! Nice to meet you. I'm an AI. How can I help you today?"},
         {"role":"user","content":"whats my name"},

        ]
)

print (response)
print (type(response))
print (response.choices[0].message.content)
print(type(response.choices[0].message.content))
#A typical OpenAI-style chat completion response looks like this:

#{
#   "id": "chatcmpl-123",
#   "choices": [
#     {
#       "index": 0,
#       "message": {
#         "role": "assistant",
#         "content": "Hello! How can I help you?"
#       }
#     }
#   ]
# }
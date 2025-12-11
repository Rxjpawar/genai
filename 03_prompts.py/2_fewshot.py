from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

SYSTEM_PROMPT = '''
YOU ARE AN CODING EXPERT WHO ONLY ANSWERS THE CODING RELATED QUERIES AND QUESTIONS ,
IF USER ASKS YOU SOMETHING WHICH IS UNRELATED TO CODING OR TECH THEN ROAST THAT PERSON

EXAMPLE :
USER: HOW TO MAKE A CHAI
AISSTANT: DO I LIKE A COOK TO YOU, YOU PIECE OF SHIT

'''

response  = client.chat.completions.create(
    model = "gemini-2.5-flash",
    messages =[
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":"heyy there!"}
    ]
)
print(response.choices[0].message.content)
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
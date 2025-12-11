from openai import OpenAI
from dotenv import load_dotenv
import os
import json


load_dotenv()
client = OpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

SYSTEM_PROMPT = f"""
    You are an helpfull AI assistant who is specialized in resolving user query.
    For the given user input, analyse the input and break down the problem step by step.

    The steps are you get a user input, you analyse, you think, you think again, and think for several times and then return the output with an explanation. 

    Follow the steps in sequence that is "analyse", "think", "output", "validate" and finally "result".

    Rules:
    1. Follow the strict JSON output as per schema.
    2. Always perform one step at a time and wait for the next input.
    3. Carefully analyse the user query,

    Output Format:
    {{ "step": "string", "content": "string" }}

    Example:
    Input: What is 2 + 2
    Output: {{ "step": "analyse", "content": "Alight! The user is interest in maths query and he is asking a basic arthematic operation" }}
    Output: {{ "step": "think", "content": "To perform this addition, I must go from left to right and add all the operands." }}
    Output: {{ "step": "output", "content": "4" }}
    Output: {{ "step": "validate", "content": "Seems like 4 is correct ans for 2 + 2" }}
    Output: {{ "step": "result", "content": "2 + 2 = 4 and this is calculated by adding all numbers" }}




"""

#    Example:
#     Input: What is 2 + 2 * 5 / 3
#     Output: {{ "step": "analyse", "content": "Alight! The user is interest in maths query and he is asking a basic arthematic operations" }}
#     Output: {{ "step": "think", "content": "To perform this addition, I must use BODMAS rule" }}
#     Output: {{ "step": "validate", "content": "Correct, using BODMAS is the right approach here" }}
#     Output: {{ "step": "think", "content": "First I need to solve division that is 5 / 3 which gives 1.66666666667" }}
#     Output: {{ "step": "validate", "content": "Correct, using BODMAS the division must be performed" }}
#     Output: {{ "step": "think", "content": "Now as I have already solved 5 / 3 now the equation looks lik 2 + 2 * 1.6666666666667" }}
#     Output: {{ "step": "validate", "content": "Yes, The new equation is absolutely correct" }}
#     Output: {{ "step": "validate", "content": "The equation now is 2 + 3.33333333333" }}
#     and so on.....

messages = [{"role": "system", "content": SYSTEM_PROMPT}]

while True:

    query = input("🐱 : ")
    if query == "exit":
        break
    messages.append({"role": "user", "content": query})

    while True:

        response = client.chat.completions.create(
            model="gemini-2.5-flash",
            response_format={"type": "json_object"},
            messages=messages,
        )

        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(response.choices[0].message.content),
            }
        )  # json.dumps takes a Python object and converts it into a JSON-formatted string.
        parsed_response = json.loads(
            response.choices[0].message.content
        )  # json.loads takes a JSON-formatted string and converts it into a Python dict/list/etc.

        if parsed_response.get("step") == "think":
            messages.append({"role": "assistant", "content": "<>"})
            continue

        if parsed_response.get("step") != "result":
            print("🧠 Thinking :", parsed_response.get("content"))
            continue

        if parsed_response.get("step") == "result":

            print("🤖 :", parsed_response.get("content"))
            break

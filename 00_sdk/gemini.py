import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key="GOOGLE_API_KEY")

model = genai.GenerativeModel("gemini-1.5-flash")

response = model.generate_content("Hello")

print(response.text)



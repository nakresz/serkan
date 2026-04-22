from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

print("API KEY:", api_key)  # debug

if api_key:
    print("API key loaded successfully ✅")
else:
    print("API key not found ❌")
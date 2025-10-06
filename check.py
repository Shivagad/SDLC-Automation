import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('API_KEY_CHATBOT')

if not api_key:
    print("Error: API_KEY_CHATBOT not found in .env file")
else:
    genai.configure(api_key=api_key)
    
    print("Available models that support generateContent:")
    print("-" * 60)
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"Model name: {m.name}")
            print(f"Display name: {m.display_name}")
            print(f"Description: {m.description}")
            print("-" * 60)
import os
from google import genai

def list_models():
    api_key = os.getenv("GEMINI_API_KEY") or "AIzaSyBl0JjZm243KSd634AywFcohhtsu_O9f-s"
    client = genai.Client(api_key=api_key)
    print("--- Available Models ---")
    try:
        for model in client.models.list():
            print(f"Name: {model.name}, Supported Actions: {model.supported_actions}")
    except Exception as e:
        print(f"Error listing models: {e}")

if __name__ == "__main__":
    list_models()

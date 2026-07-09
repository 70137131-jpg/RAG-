import os
from dotenv import load_dotenv
import requests

load_dotenv()

provider = os.getenv("LLM_PROVIDER", "").lower().strip()
model = os.getenv("LLM_MODEL", "")
if not provider:
    provider = "gemini" if "gemini" in model.lower() else "openrouter"

try:
    if provider == "gemini":
        from google import genai

        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        for model_info in client.models.list():
            print(model_info.name)
    elif provider == "openrouter":
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY', '')}"},
            timeout=30,
        )
        response.raise_for_status()
        for model_info in response.json().get("data", []):
            print(model_info.get("id"))
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
except Exception as e:
    print("Error listing models:", e)

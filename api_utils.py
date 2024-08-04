from functools import lru_cache
from groq import Groq
import openai
import os
from dotenv import load_dotenv

# --- Global Settings and Constants ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


@lru_cache(maxsize=None)
def get_api_client(provider: str):
    """Returns the appropriate API client based on the selected provider."""
    try:
        if provider == "Groq":
            return Groq(api_key=GROQ_API_KEY)
        elif provider == "OpenAI":
            return openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
        elif provider == "Google":
            return None  # Gemini doesn't use a client object
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    except Exception as e:
        # Handle error - you might want to log this 
        print(f"Error initializing {provider} client: {e}")
        return None 
# api/__init__.py

import os
from typing import Dict, List, AsyncGenerator
from .groq import GroqClient
from .openai import OpenAIClient
from .google import GoogleClient
from .mistral import MistralClient

async def get_api_client(provider: str):
    if provider == "Groq":
        return GroqClient(os.getenv("GROQ_API_KEY"))
    elif provider == "OpenAI":
        return OpenAIClient(os.getenv("OPENAI_API_KEY"))
    elif provider == "Google":
        return GoogleClient(os.getenv("GEMINI_API_KEY"))
    elif provider == "Mistral":
        return MistralClient(os.getenv("MISTRAL_API_KEY"))
    else:
        raise ValueError(f"Unsupported provider: {provider}")

async def async_stream_llm_response(
    client: any,
    params: Dict,
    messages: List[Dict],
    provider: str,
    voice: str = "alloy"
) -> AsyncGenerator[str, None]:
    try:
        async for chunk in client.stream_response(params, messages):
            yield chunk
    except Exception as e:
        yield f"Error in {provider} API call: {str(e)}"
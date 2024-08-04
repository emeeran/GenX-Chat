import asyncio
import aiohttp
import logging
import json
from typing import Dict, List, Any 
import os
from dotenv import load_dotenv

# --- Global Settings and Constants ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# --- Gemini Function Declarations ---
GEMINI_FUNCTIONS = [
    {
        "name": "get_weather",
        "description": "Get weather information for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The name of the city or location.",
                },
                "date": {
                    "type": "string",
                    "description": "The date for which to retrieve the weather (format: YYYY-MM-DD).",
                    "optional": True,
                },
            },
            "required": ["location"],
        },
    },
    # ... (You can add more Gemini functions here)
]

logger = logging.getLogger(__name__)

async def async_stream_gemini_response(
    params: Dict[str, Any], 
    messages: List[Dict[str, str]],
) -> str:
    """Streams the Google Gemini LLM response asynchronously."""
    try:
        request_body = {
            "model": params["model"],
            "prompt": {
                "text": messages[-1]["content"],
            },
            "tool_code": {"function_declarations": GEMINI_FUNCTIONS},
        }

        headers = {
            "Authorization": f"Bearer {GEMINI_API_KEY}",
            "Content-Type": "application/json",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.google.com/v1/generative/text:generateText",
                json=request_body,
                headers=headers,
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if "candidates" in data and data["candidates"]:
                        candidate = data["candidates"][0]
                        if "function_call" in candidate:
                            function_call = candidate["function_call"]
                            function_name = function_call["name"]
                            function_args = json.loads(function_call.get("arguments", "{}"))
                            yield f"Function Call: {function_name}\nArguments: {function_args}"
                        else:
                            yield candidate["output"]
                    else:
                        logger.error(f"Google API response error: {data}")
                        yield f"Error in Google API response: {data}"
                else:
                    error_text = await response.text()
                    logger.error(f"Google API Error: {response.status} - {error_text}")
                    yield f"Google API Error: {response.status} - {error_text}"

    except Exception as e:
        logger.error(f"Error in Google API call: {str(e)}")
        yield f"Error in Google API call: {str(e)}"
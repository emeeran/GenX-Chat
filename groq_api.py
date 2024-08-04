import asyncio
import aiohttp
import logging
import json
import os
from dotenv import load_dotenv
from typing import Dict, List, Any  # Import Dict, List, Any

# --- Global Settings and Constants ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

logger = logging.getLogger(__name__)

async def async_stream_groq_response(
    client,
    params: Dict[str, Any],  # Now correctly defined
    messages: List[Dict[str, str]],  # Now correctly defined
) -> str:
    """Streams the Groq LLM response asynchronously."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": params["model"],
                    "messages": messages,
                    "max_tokens": params.get("max_tokens"),
                    "temperature": params.get("temperature"),
                    "top_p": params.get("top_p"),
                    "stream": True,
                },
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Groq API Error: {response.status} - {error_text}")
                    yield f"Groq API Error: {response.status} - {error_text}"
                    return

                async for line in response.content:
                    if line.strip() and line.startswith(b"data: "):
                        try:
                            json_str = line[6:].decode("utf-8").strip()
                            if json_str != "[DONE]":
                                chunk = json.loads(json_str)
                                if (
                                    "choices" in chunk
                                    and chunk["choices"]
                                    and "delta" in chunk["choices"][0]
                                    and "content" in chunk["choices"][0]["delta"]
                                ):
                                    yield chunk["choices"][0]["delta"]["content"]
                        except json.JSONDecodeError as json_err:
                            logger.error(f"JSON decode error: {json_err}. Raw line: {line}")
                        except Exception as e:
                            logger.error(f"Error processing line: {e}. Raw line: {line}")
    except Exception as e:
        logger.error(f"Error in Groq API call: {str(e)}")
        yield f"Error in Groq API call: {str(e)}"
# api/groq.py

import json
import logging
from typing import Dict, List, AsyncGenerator
import aiohttp
from .base import BaseAPIClient

logger = logging.getLogger(__name__)

class GroqClient(BaseAPIClient):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"

    async def stream_response(self, params: Dict, messages: List[Dict]) -> AsyncGenerator[str, None]:
        async with aiohttp.ClientSession(headers={"Authorization": f"Bearer {self.api_key}"}) as session:
            async with session.post(
                self.base_url,
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
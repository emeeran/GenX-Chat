# api/google.py

import json
import logging
from typing import Dict, List, AsyncGenerator
import aiohttp
from .base import BaseAPIClient
from config import GEMINI_FUNCTIONS

logger = logging.getLogger(__name__)

class GoogleClient(BaseAPIClient):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.google.com/v1/generative/text:generateText"

    async def stream_response(self, params: Dict, messages: List[Dict]) -> AsyncGenerator[str, None]:
        try:
            request_body = {
                "model": params["model"],
                "prompt": {
                    "text": messages[-1]["content"],
                },
                "tool_code": {"function_declarations": GEMINI_FUNCTIONS},
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.base_url,
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
# api/openai.py

import logging
from typing import Dict, List, AsyncGenerator
from openai import AsyncOpenAI
from .base import BaseAPIClient

logger = logging.getLogger(__name__)

class OpenAIClient(BaseAPIClient):
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)

    async def stream_response(self, params: Dict, messages: List[Dict]) -> AsyncGenerator[str, None]:
        try:
            response = await self.client.chat.completions.create(
                model=params["model"],
                messages=messages,
                max_tokens=params["max_tokens"],
                temperature=params["temperature"],
                top_p=params["top_p"],
                stream=True,
            )
            async for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"Error in OpenAI API call: {str(e)}")
            yield f"Error in OpenAI API call: {str(e)}"
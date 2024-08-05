import os
import base64
import asyncio
import logging
import streamlit as st  # Import streamlit
from typing import Dict, List, Any 
from dotenv import load_dotenv

from utils import translate_text  # Import the translate_text function

# --- Global Settings and Constants ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

logger = logging.getLogger(__name__)

async def async_stream_openai_response(
    client,
    params: Dict[str, Any], 
    messages: List[Dict[str, str]], 
    voice: str = "alloy",
):
    """Streams the OpenAI LLM response asynchronously."""
    try:
        if st.session_state.language != "English":
            assistant_response = translate_text(
                messages[-1]["content"], st.session_state.language
            )
            messages[-1]["content"] = assistant_response

        response = await client.chat.completions.create(
            model=params["model"],
            messages=[{"role": "assistant", "content": messages[-1]["content"]}],
            max_tokens=params["max_tokens"],
            temperature=params["temperature"],
            top_p=params["top_p"],
        )

        yield response.choices[0].message.content

    except Exception as e:
        logger.error(f"Error in OpenAI API call: {str(e)}")
        yield f"Error in OpenAI API call: {str(e)}"
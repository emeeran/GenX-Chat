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
) -> str:
    """Streams the OpenAI LLM response asynchronously."""
    try:
        if st.session_state.language != "English":
            assistant_response = translate_text(
                messages[-1]["content"], st.session_state.language
            )
            messages[-1]["content"] = assistant_response

        response = await client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=messages[-1]["content"],
        )

        audio_file = "temp_audio.mp3"
        with open(audio_file, "wb") as f:
            f.write(response.content)

        with open(audio_file, "rb") as f:
            audio_bytes = f.read()
        st.session_state.audio_base64 = base64.b64encode(audio_bytes).decode()

        os.remove(audio_file)
        yield messages[-1]["content"]

    except Exception as e:
        logger.error(f"Error in OpenAI API call: {str(e)}")
        yield f"Error in OpenAI API call: {str(e)}"
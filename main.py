# main.py

import os
import asyncio
import logging
from typing import Dict, Any

import streamlit as st
from dotenv import load_dotenv

from config import (
    PROVIDER_OPTIONS, MODEL_OPTIONS, VOICE_OPTIONS, LANGUAGE_OPTIONS,
    SUMMARIZATION_TYPES, COLOR_SCHEMES, MAX_CHAT_HISTORY_LENGTH, DB_PATH,
    DEFAULT_PROVIDER, DEFAULT_MODEL_PARAMS, COST_PER_TOKEN
)
from persona import PERSONAS
from content_type import CONTENT_TYPES
from api import get_api_client, async_stream_llm_response
from utils.file_processing import process_uploaded_file
from utils.text_processing import word_count, translate_text
from utils.audio import text_to_speech
from database.chat_history import (
    create_database, save_chat_history_to_db, load_chat_history_from_db,
    get_saved_chat_names, delete_chat
)
from ui.components import setup_sidebar, setup_main_page
import os
import asyncio
import logging
from typing import Dict, Any

import streamlit as st
from dotenv import load_dotenv

from config import (
    PROVIDER_OPTIONS, PERSONAS, CONTENT_TYPES, VOICE_OPTIONS,
    MAX_CHAT_HISTORY_LENGTH, DB_PATH
)
from api import get_api_client, async_stream_llm_response
from utils.file_processing import process_uploaded_file
from utils.text_processing import word_count, translate_text
from utils.audio import text_to_speech
from database.chat_history import (
    create_database, save_chat_history_to_db, load_chat_history_from_db,
    get_saved_chat_names, delete_chat
)
from ui.components import setup_sidebar, setup_main_page

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
def initialize_session_state():
    default_values = {
        "messages": [],
        "audio_base64": "",
        "file_content": "",
        "persona": "Default",
        "model_params": {
            "model": "google/gemini-1.0-pro" if os.getenv("GEMINI_API_KEY") else "llama-3.1-70b-versatile",
            "max_tokens": 1024,
            "temperature": 1.0,
            "top_p": 1.0,
        },
        "total_tokens": 0,
        "total_cost": 0,
        "enable_audio": False,
        "language": "English",
        "voice": "alloy",
        "content_creation_mode": False,
        "show_summarization": False,
        "summarization_type": "Main Takeaways",
        "content_type": "Short Story",
        "provider": "Google" if os.getenv("GEMINI_API_KEY") else "Groq" if os.getenv("GROQ_API_KEY") else "OpenAI" if os.getenv("OPENAI_API_KEY") else None,
        "color_scheme": "Light",
    }
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Main chat processing function
async def process_chat_input(prompt: str, client: Any) -> None:
    try:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        if st.session_state.file_content:
            prompt = f"Based on the uploaded file content, {prompt}\n\nFile content: {st.session_state.file_content[:4000]}..."

        messages = [
            {"role": "system", "content": PERSONAS[st.session_state.persona]},
            *st.session_state.messages[-MAX_CHAT_HISTORY_LENGTH:],
            {"role": "user", "content": prompt},
        ]

        with st.chat_message("user"):
            st.markdown(prompt)

        full_response = ""
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            async for chunk in async_stream_llm_response(
                client,
                st.session_state.model_params,
                messages,
                st.session_state.provider,
                st.session_state.voice,
            ):
                if chunk.startswith("API Error:"):
                    message_placeholder.error(chunk)
                    return
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

        st.session_state.messages.extend([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": full_response}
        ])

        if st.session_state.enable_audio and full_response.strip():
            if st.session_state.provider == "Groq":
                text_to_speech(full_response, st.session_state.language)
            st.audio(f"data:audio/mp3;base64,{st.session_state.audio_base64}", format="audio/mp3")

        st.session_state.total_tokens += len(full_response.split())
        st.session_state.total_cost += len(full_response.split()) * 0.0001

        if st.session_state.content_creation_mode:
            content_type = CONTENT_TYPES[st.session_state.content_type]
            generated_content = await create_content(prompt, content_type, client)
            with st.chat_message("assistant"):
                st.markdown(f"## Generated {st.session_state.content_type}:\n\n{generated_content}")

        if st.session_state.show_summarization:
            text_to_summarize = st.session_state.file_content or prompt
            summary = await summarize_text(text_to_summarize, st.session_state.summarization_type, client)
            with st.chat_message("assistant"):
                st.markdown(f"## Summary ({st.session_state.summarization_type}):\n\n{summary}")

    except ValueError as ve:
        st.error(f"Invalid input: {str(ve)}")
    except Exception as e:
        logger.error(f"Unexpected error in process_chat_input: {str(e)}", exc_info=True)
        st.error(f"An unexpected error occurred. Please try again later.")

# Content creation function
async def create_content(prompt: str, content_type: str, client: Any) -> str:
    full_prompt = f"Write a {content_type} based on this prompt: {prompt}"
    messages = [{"role": "user", "content": full_prompt}]
    generated_content = ""
    async for chunk in async_stream_llm_response(
        client,
        st.session_state.model_params,
        messages,
        st.session_state.provider,
        st.session_state.voice,
    ):
        generated_content += chunk
    return generated_content

# Text summarization function
async def summarize_text(text: str, summary_type: str, client: Any) -> str:
    full_prompt = f"Please provide a {summary_type} of the following text: {text}"
    messages = [{"role": "user", "content": full_prompt}]
    summary = ""
    async for chunk in async_stream_llm_response(
        client,
        st.session_state.model_params,
        messages,
        st.session_state.provider,
        st.session_state.voice,
    ):
        summary += chunk
    return summary

# Main function
async def main() -> None:
    st.set_page_config(page_title="GenX-Chat", page_icon="💬", layout="wide")
    
    initialize_session_state()

    await create_database()

    if 'saved_chats' not in st.session_state:
        st.session_state.saved_chats = await get_saved_chat_names()

    setup_sidebar()
    setup_main_page()

    if hasattr(st.session_state, 'load_chat'):
        await load_chat_history_from_db(st.session_state.load_chat)
        del st.session_state.load_chat

    if hasattr(st.session_state, 'save_chat'):
        await save_chat_history_to_db(st.session_state.save_chat)
        st.success(f"Chat '{st.session_state.save_chat}' saved successfully.")
        del st.session_state.save_chat

    if hasattr(st.session_state, 'delete_chat'):
        await delete_chat(st.session_state.delete_chat)
        st.success(f"Chat '{st.session_state.delete_chat}' deleted successfully.")
        del st.session_state.delete_chat

    client = await get_api_client(st.session_state.provider)
    if client is None and st.session_state.provider != "Google":
        st.error(f"Failed to initialize {st.session_state.provider} client. Please check your API key.")
        return

    prompt = st.chat_input("Enter your message:")
    if prompt:
        await process_chat_input(prompt, client)

    # Close the client session if it's an aiohttp.ClientSession
    if hasattr(client, 'close'):
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
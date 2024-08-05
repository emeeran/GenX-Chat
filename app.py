import os
import base64
import logging
import asyncio
import json
from typing import List, Dict, Any
from functools import lru_cache

import streamlit as st
from dotenv import load_dotenv
from gtts import gTTS
from PIL import Image

# API and utility imports
from api_utils import get_api_client
from groq_api import async_stream_groq_response
from openai_api import async_stream_openai_response
from google_gemini_api import async_stream_gemini_response
from utils import (
    validate_prompt, process_uploaded_file, text_to_speech, translate_text,
    update_token_count, export_chat, create_database, save_chat_history_to_db,
    load_chat_history_from_db, get_saved_chat_names, delete_chat, create_content,
    summarize_text, get_model_options, get_max_token_limit, word_count,
    initialize_session_state, reset_current_chat
)

# Custom module imports
from content_type import CONTENT_TYPES
from persona import PERSONAS

# Global settings and constants
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

MAX_CHAT_HISTORY_LENGTH = 50
DB_PATH = "chat_history.db"

VOICE_OPTIONS = {
    "OpenAI": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
    "gTTS": ["en", "ta", "hi"],
}

PROVIDER_OPTIONS = ["Google", "Groq", "OpenAI"]

async def async_stream_llm_response(
    client, params: Dict[str, Any], messages: List[Dict[str, str]], provider: str, voice: str = "alloy"
):
    try:
        if provider == "Groq":
            async for chunk in async_stream_groq_response(client, params, messages):
                yield chunk
        elif provider == "OpenAI":
            async for chunk in async_stream_openai_response(client, params, messages, voice):
                yield chunk
        elif provider == "Google":
            async for chunk in async_stream_gemini_response(params, messages):
                yield chunk
        else:
            yield f"Error: Unsupported provider: {provider}"
    except Exception as e:
        logger.error(f"Error in async_stream_llm_response: {str(e)}", exc_info=True)
        yield f"API Error: {str(e)}"

async def process_chat_input(prompt: str, client):
    try:
        validate_prompt(prompt)

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

        update_token_count(len(full_response.split()))

        if st.session_state.content_creation_mode:
            content_type = CONTENT_TYPES[st.session_state.content_type]
            generated_content = await create_content(prompt, content_type)
            with st.chat_message("assistant"):
                st.markdown(f"## Generated {st.session_state.content_type}:\n\n{generated_content}")

        if st.session_state.show_summarization:
            text_to_summarize = st.session_state.file_content or prompt
            summary = await summarize_text(text_to_summarize, st.session_state.summarization_type)
            with st.chat_message("assistant"):
                st.markdown(f"## Summary ({st.session_state.summarization_type}):\n\n{summary}")

    except ValueError as ve:
        st.error(f"Invalid input: {str(ve)}")
    except Exception as e:
        logger.error(f"Unexpected error in process_chat_input: {str(e)}", exc_info=True)
        st.error(f"An unexpected error occurred. Please try again later.")

def setup_sidebar():
    with st.sidebar:
        st.markdown("<h2 style='text-align: center;color: #6ca395;'>Select Provider</h2>", unsafe_allow_html=True)
        
        available_providers = [p for p in PROVIDER_OPTIONS if os.getenv(f"{p.upper()}_API_KEY")]
        
        if not available_providers:
            st.error("No API keys are set. Please set at least one API key in your .env file.")
            st.stop()

        selected_provider = st.selectbox(
            "Test", available_providers, label_visibility="collapsed",
            format_func=lambda x: "Select Provider" if x == "" else x,
        )
        st.session_state.provider = selected_provider

        st.markdown("<h2 style='text-align: center;'>⚙️ Settings</h2>", unsafe_allow_html=True)

        with st.expander("Chat Settings", expanded=True):
            saved_chats = st.session_state.get('saved_chats', [])
            selected_chat = st.selectbox("Load Chat History", options=[""] + saved_chats)
            if selected_chat:
                st.session_state.load_chat = selected_chat

            col4, col5, col6 = st.columns(3)
            with col4:
                if st.button("Retry"):
                    st.rerun()
            with col5:
                if st.button("New"):
                    reset_current_chat()
            with col6:
                if st.button("Delete"):
                    if selected_chat:
                        st.session_state.delete_chat = selected_chat
                        st.rerun()

            col1, col2 = st.columns([2, 1])
            with col1:
                chat_name_input = st.text_input("Enter a name for this chat:", max_chars=50, label_visibility="collapsed", placeholder="Chat Name", help="Type a name for your chat")
            with col2:
                if st.button("Save"):
                    if chat_name_input:
                        st.session_state.save_chat = chat_name_input
                        st.rerun()

            st.button("Reset", on_click=reset_current_chat)

        with st.expander("Model"):
            model_options = get_model_options(st.session_state.provider)
            st.session_state.model_params["model"] = st.selectbox("Choose Model:", options=model_options)

            max_token_limit = get_max_token_limit(st.session_state.model_params["model"])
            st.session_state.model_params["max_tokens"] = st.slider("Max Tokens:", min_value=1, max_value=max_token_limit, value=min(1024, max_token_limit), step=1)
            st.session_state.model_params["temperature"] = st.slider("Temperature:", 0.0, 2.0, 1.0, 0.1)
            st.session_state.model_params["top_p"] = st.slider("Top-p:", 0.0, 1.0, 1.0, 0.1)

        with st.expander("Persona"):
            persona_options = list(PERSONAS.keys())
            st.session_state.persona = st.selectbox("Select Persona:", options=persona_options, index=persona_options.index("Default"))
            st.text_area("Persona Description:", value=PERSONAS[st.session_state.persona], height=100, disabled=True)

        with st.expander("Audio & Language"):
            st.session_state.enable_audio = st.checkbox("Enable Audio Response", value=False)
            language_options = ["English", "Tamil", "Hindi"]
            st.session_state.language = st.selectbox("Select Language:", language_options)
            voice_options = VOICE_OPTIONS["OpenAI"] if st.session_state.provider == "OpenAI" else VOICE_OPTIONS["gTTS"]
            st.session_state.voice = st.selectbox(f"Select {'Voice' if st.session_state.provider == 'OpenAI' else 'Language Code'}:", voice_options)

        with st.expander("File Upload"):
            uploaded_file = st.file_uploader("Upload a file", type=["pdf", "docx", "txt", "md", "jpg", "jpeg", "png"])
            if uploaded_file:
                try:
                    st.session_state.file_content = process_uploaded_file(uploaded_file)
                    st.success("File processed successfully")
                except Exception as e:
                    st.error(f"Error processing file: {e}")

        with st.expander("Summarize"):
            st.session_state.show_summarization = st.checkbox("Enable Summarization", value=False)
            if st.session_state.show_summarization:
                st.session_state.summarization_type = st.selectbox(
                    "Summarization Type:",
                    ["Main Takeaways", "Main points bulleted", "Concise Summary", "Executive Summary"],
                )

        with st.expander("Content Generation"):
            st.session_state.content_creation_mode = st.checkbox("Enable Content Creation Mode", value=False)
            if st.session_state.content_creation_mode:
                st.session_state.content_type = st.selectbox("Select Content Type:", list(CONTENT_TYPES.keys()))

        with st.expander("Export"):
            export_format = st.selectbox("Export Format", ["md", "pdf"])
            st.button("Export Chat", on_click=lambda: export_chat(export_format))

        st.session_state.color_scheme = st.selectbox("Color Scheme", ["Light", "Dark"])
        if st.session_state.color_scheme == "Dark":
            st.markdown(
                """
                <style>
                .stApp {
                    background-color: #1E1E1E;
                    color: #FFFFFF;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

async def main():
    initialize_session_state()

    await create_database()

    if 'saved_chats' not in st.session_state:
        st.session_state.saved_chats = await get_saved_chat_names()

    setup_sidebar()

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

    client = get_api_client(st.session_state.provider)
    if client is None and st.session_state.provider != "Google":
        st.error(f"Failed to initialize {st.session_state.provider} client. Please check your API key.")
        return

    st.markdown('<h1 style="text-align: center; color: #6ca395;">GenX-Chat 💬</h1>', unsafe_allow_html=True)

    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages[-MAX_CHAT_HISTORY_LENGTH:]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    prompt = st.chat_input("Enter your message:")
    if prompt:
        await process_chat_input(prompt, client)

    total_words = sum(word_count(msg["content"]) for msg in st.session_state.messages)
    st.sidebar.metric("Total Words", total_words)
    st.sidebar.metric("Total Tokens", st.session_state.total_tokens)
    st.sidebar.metric("Estimated Cost", f"${st.session_state.total_cost:.4f}")

if __name__ == "__main__":
    st.set_page_config(page_title="GenX-Chat", page_icon="💬", layout="wide")
    asyncio.run(main())
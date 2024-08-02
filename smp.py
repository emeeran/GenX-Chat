import os
import base64
from gtts import gTTS
import PyPDF2
import docx
import pytesseract
from deep_translator import GoogleTranslator
from datetime import datetime
from fpdf import FPDF
from dotenv import load_dotenv
import logging
import asyncio
import aiohttp
import streamlit as st
import json
from PIL import Image
from typing import List, Dict, Any
from content_type import CONTENT_TYPES
from persona import PERSONAS
from groq import Groq


# --- Global Settings and Constants ---
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
if API_KEY is None:
    st.error("GROQ_API_KEY environment variable not set. Please set it in your .env file.")
    st.stop()

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

MAX_CHAT_HISTORY_LENGTH = 50

# --- Utility Functions ---
def get_groq_client(api_key: str):
    """Returns a Groq client instance."""
    try:
        return Groq(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to initialize Groq client: {str(e)}")
        return None

async def async_stream_llm_response(client: Groq, params: Dict[str, Any], messages: List[Dict[str, str]]):
    """Streams the LLM response from the Groq API."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json"
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
                    logger.error(f"API Error: {response.status} - {error_text}")
                    yield f"API Error: {response.status} - {error_text}"
                    return

                async for line in response.content:
                    if line.strip() and line.startswith(b"data: "):
                        try:
                            json_str = line[6:].decode('utf-8').strip()
                            if json_str != "[DONE]":
                                chunk = json.loads(json_str)
                                if "choices" in chunk and chunk["choices"] and "delta" in chunk["choices"][0] and "content" in chunk["choices"][0]["delta"]:
                                    yield chunk["choices"][0]["delta"]["content"]
                        except json.JSONDecodeError as json_err:
                            logger.error(f"JSON decode error: {json_err}. Raw line: {line}")
                        except Exception as e:
                            logger.error(f"Error processing line: {e}. Raw line: {line}")

    except Exception as e:
        logger.error(f"Error in API call: {str(e)}")
        yield f"Error in API call: {str(e)}"

def validate_prompt(prompt: str):
    """Validates the user prompt."""
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty")

def process_uploaded_file(uploaded_file):
    """Processes uploaded files and extracts content."""
    file_handlers = {
        "application/pdf": lambda f: " ".join(page.extract_text() for page in PyPDF2.PdfReader(f).pages),
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": lambda f: " ".join(paragraph.text for paragraph in docx.Document(f).paragraphs),
        "text/plain": lambda f: f.getvalue().decode("utf-8"),
        "text/markdown": lambda f: f.getvalue().decode("utf-8"),
        "image/jpeg": lambda f: pytesseract.image_to_string(Image.open(f)),
        "image/png": lambda f: pytesseract.image_to_string(Image.open(f)),
    }
    for file_type, handler in file_handlers.items():
        if uploaded_file.type.startswith(file_type):
            return handler(uploaded_file)
    raise ValueError("Unsupported file type")

def text_to_speech(text: str, lang: str):
    """Converts text to speech."""
    lang_map = {"english": "en", "tamil": "ta", "hindi": "hi"}
    lang_code = lang_map.get(lang.lower(), "en")
    tts = gTTS(text=text, lang=lang_code)
    audio_file = "temp_audio.mp3"
    tts.save(audio_file)
    with open(audio_file, "rb") as f:
        audio_bytes = f.read()
    os.remove(audio_file)
    st.session_state.audio_base64 = base64.b64encode(audio_bytes).decode()

def translate_text(text: str, target_lang: str) -> str:
    """Translates text."""
    if target_lang == "English":
        return text
    translator = GoogleTranslator(source="auto", target=target_lang)
    return translator.translate(text)

def update_token_count(tokens: int):
    """Updates the token count."""
    st.session_state.total_tokens += tokens
    st.session_state.total_cost += tokens * 0.0001  # Assuming a cost per token

def export_chat(format: str):
    """Exports the chat history."""
    chat_history = "\n\n".join([f"**{m['role'].capitalize()}:** {m['content']}" for m in st.session_state.messages])
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"chat_exports/chat_history_{timestamp}.{format}"
    os.makedirs("chat_exports", exist_ok=True)

    if format == "md":
        with open(filename, "w") as f:
            f.write(chat_history)
        st.download_button("Download Markdown", filename, file_name=filename)
    elif format == "pdf":
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, chat_history)
        pdf.output(filename)
        st.download_button("Download PDF", filename, file_name=filename)

# --- Chat History Management ---
def save_chat_history():
    """Saves the chat history."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    st.session_state.chat_histories.append({"name": f"Chat {timestamp}", "messages": st.session_state.messages.copy()})

def load_chat_history(selected_history: str):
    """Loads a chat history."""
    for history in st.session_state.chat_histories:
        if history["name"] == selected_history:
            st.session_state.messages = history["messages"].copy()
            break

# --- Content Creation (Uses LLM) ---
async def create_content(prompt: str, content_type: str) -> str:
    """Generates content using the LLM."""
    full_prompt = f"Write a {content_type} based on this prompt: {prompt}"
    generated_content = ""
    async for chunk in async_stream_llm_response(
        get_groq_client(API_KEY),
        st.session_state.model_params,
        [{"role": "user", "content": full_prompt}],
    ):
        generated_content += chunk
    return generated_content

# --- Summarization (Uses LLM) ---
async def summarize_text(text: str, summary_type: str) -> str:
    """Summarizes text using the LLM."""
    full_prompt = f"Please provide a {summary_type} of the following text: {text}"
    summary = ""
    async for chunk in async_stream_llm_response(
        get_groq_client(API_KEY),
        st.session_state.model_params,
        [{"role": "user", "content": full_prompt}],
    ):
        summary += chunk
    return summary

# --- Initialize Session State ---
def initialize_session_state():
    """Initializes session state."""
    default_values = {
        "messages": [],
        "audio_base64": "",
        "file_content": "",
        "chat_histories": [],
        "persona": "Default",
        "model_params": {"model": "llama-3.1-70b-versatile", "max_tokens": 11024, "temperature": 1.0, "top_p": 1.0},
        "total_tokens": 0,
        "total_cost": 0,
        "enable_audio": False,
        "language": "English",
        "content_creation_mode": False,
        "show_summarization": False,
        "summarization_type": "Main Takeaways",
        "content_type": "Short Story",
    }
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- Main Streamlit Application ---
def main():
    """Main Streamlit app function."""
    initialize_session_state()
    client = get_groq_client(API_KEY)
    if client is None:
        st.error("Failed to initialize Groq client. Please check your API key and try again.")
        return

    st.set_page_config(
        page_title="GenX-Chat",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stSidebar.expanded {
        padding-left: 0; /* Reset the padding to align the sidebar to the left */
        margin-left: 0; /* Reset the margin to align the sidebar to the left */
    }
    .stChatMessage {
        max-width: 80%;
    }
    @media (max-width: 768px) {
        .stChatMessage {
            max-width: 90%;
        }
    }
    </style>
    """, unsafe_allow_html=True)


    st.markdown('<h1 style="text-align: center; color: #6ca395;">GenX-Chat 💬</h1>', unsafe_allow_html=True)

    # --- Sidebar ---
    with st.sidebar:
        # Chat Settings 
        with st.expander("Chat Settings", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.button("Reset All", on_click=lambda: st.session_state.clear())
            with col2:
                st.button("Save Chat History", on_click=save_chat_history)

            chat_history_names = [history["name"] for history in st.session_state.chat_histories]
            selected_history = st.selectbox("Load Chat History", options=[""] + chat_history_names)
            if selected_history:
                load_chat_history(selected_history)

        # Model Settings
        with st.expander("Model"):
            st.session_state.model_params["model"] = st.selectbox(
                "Choose Model:",
                options=[
                    "llama-3.1-70b-versatile",
                    "llama-3.1-405b-reasoning",
                    "llama-3.1-8b-instant",
                    "llama3-groq-70b-8192-tool-use-preview",
                    "llama3-70b-8192",
                    "mixtral-8x7b-32768",
                    "gemma2-9b-it",
                    "whisper-large-v3"
                ],
            )

            max_token_limit = 4096
            if st.session_state.model_params["model"] == "mixtral-8x7b-32768":
                max_token_limit = 32768
            elif st.session_state.model_params["model"] == "llama-3.1-70b-versatile-131072":
                max_token_limit = 131072
            elif st.session_state.model_params["model"] == "gemma2-9b-it":
                max_token_limit = 8192

            st.session_state.model_params["max_tokens"] = st.slider(
                "Max Tokens:", min_value=1, max_value=max_token_limit, value=min(1024, max_token_limit), step=1
            )
            st.session_state.model_params["temperature"] = st.slider("Temperature:", 0.0, 2.0, 1.0, 0.1)
            st.session_state.model_params["top_p"] = st.slider("Top-p:", 0.0, 1.0, 1.0, 0.1)

        # Persona Settings
        with st.expander("Persona"):
            persona_options = list(PERSONAS.keys())
            st.session_state.persona = st.selectbox("Select Persona:", options=persona_options, index=persona_options.index("Default"))
            st.text_area("Persona Description:", value=PERSONAS[st.session_state.persona], height=100, disabled=True)

        # Audio & Language Settings
        with st.expander("Audio & Language"):
            st.session_state.enable_audio = st.checkbox("Enable Audio Response", value=False)
            st.session_state.language = st.selectbox("Select Language:", ["English", "Tamil", "Hindi"])

        # Content Generation
        with st.expander("Content Generation"):
            st.session_state.content_creation_mode = st.checkbox("Enable Content Creation Mode", value=False)
            if st.session_state.content_creation_mode:
                st.session_state.content_type = st.selectbox("Select Content Type:", list(CONTENT_TYPES.keys()))

        # Summarization 
        with st.expander("Summarize"):
            st.session_state.show_summarization = st.checkbox("Enable Summarization", value=False)
            if st.session_state.show_summarization:
                st.session_state.summarization_type = st.selectbox(
                    "Summarization Type:",
                    ["Main Takeaways", "Main points bulleted", "Concise Summary", "Executive Summary"],
                )

        # Export Chat
        with st.expander("Export"):
            export_format = st.selectbox("Export Format", ["md", "pdf"])
            st.button("Export Chat", on_click=lambda: export_chat(export_format))

        # File Upload
        with st.expander("File Upload"):
            uploaded_file = st.file_uploader("Upload a file", type=["pdf", "docx", "txt", "md", "jpg", "jpeg", "png"])
            if uploaded_file:
                try:
                    st.session_state.file_content = process_uploaded_file(uploaded_file)
                    st.success("File processed successfully")
                except Exception as e:
                    st.error(f"Error processing file: {e}")

    # --- Main Chat Interface ---
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages[-MAX_CHAT_HISTORY_LENGTH:]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # --- Prompt Input ---
    prompt = st.chat_input("Enter your message:")
    if prompt:
        asyncio.run(process_chat_input(prompt, client))

async def process_chat_input(prompt: str, client: Groq):
    """Processes chat input, gets a response, and manages chat history."""
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
            async for chunk in async_stream_llm_response(client, st.session_state.model_params, messages):
                if chunk.startswith("API Error:") or chunk.startswith("Error in API call:"):
                    message_placeholder.error(chunk)
                    return
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        if st.session_state.enable_audio and full_response.strip():
            text_to_speech(full_response, st.session_state.language)
            st.audio(f"data:audio/mp3;base64,{st.session_state.audio_base64}", format="audio/mp3")

        update_token_count(len(full_response.split()))

        # Handle content creation
        if st.session_state.content_creation_mode:
            content_type = CONTENT_TYPES[st.session_state.content_type]
            generated_content = await create_content(prompt, content_type)
            with st.chat_message("assistant"):
                st.markdown(f"## Generated {st.session_state.content_type}:\n\n{generated_content}")

        # Handle summarization
        if st.session_state.show_summarization:
            text_to_summarize = st.session_state.file_content if st.session_state.file_content else prompt
            summary = await summarize_text(text_to_summarize, st.session_state.summarization_type)
            with st.chat_message("assistant"):
                st.markdown(f"## Summary ({st.session_state.summarization_type}):\n\n{summary}")

    except ValueError as ve:
        st.error(f"Invalid input: {str(ve)}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        logger.error(f"Unexpected error in process_chat_input: {str(e)}", exc_info=True)

if __name__ == "__main__":
    if not API_KEY:
        st.error("GROQ_API_KEY is not set. Please check your .env file.")
        st.stop()
    main()

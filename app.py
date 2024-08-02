import os
import base64
from gtts import gTTS
import PyPDF2
import docx
import markdown
from PIL import Image
import pytesseract
from deep_translator import GoogleTranslator
from datetime import datetime
import json
from fpdf import FPDF
from dotenv import load_dotenv
from groq import Groq
import logging
from typing import List, Dict, Any
import asyncio
import aiohttp
import streamlit as st
from streamlit_ace import st_ace
import html

from persona import PERSONAS

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
if API_KEY is None:
    st.error("GROQ_API_KEY environment variable not set. Please set it in your .env file.")
    st.stop()

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

MAX_CHAT_HISTORY_LENGTH = 50

# --- Utility Functions ---
def get_groq_client(api_key: str) -> Groq:
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
                    if line.strip():
                    if line.strip():
                        try:
                            if line.startswith(b"data: "):
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
    """Exports the chat history in the specified format."""
    chat_history_text = "\n\n".join([f"**{m['role'].capitalize()}:** {m['content']}" for m in st.session_state.messages])
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    if format == "json":
        filename = f"chat_history_{timestamp}.json"
        data = json.dumps(st.session_state.messages, indent=4)
        return filename, data

    elif format == "md":
        filename = f"chat_history_{timestamp}.md"
        return filename, chat_history_text

    elif format == "pdf":
        filename = f"chat_history_{timestamp}.pdf"
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, chat_history_text)
        pdf.output(filename)
        with open(filename, "rb") as f:
            return filename, f.read()

# --- Chat History Management ---
def save_chat_history():
    """Saves the chat history to a JSON file in the chat_history directory."""
    os.makedirs("chat_history", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"chat_history/chat_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(st.session_state.messages, f, indent=4)
    st.success(f"Chat history saved as {filename}")
    return filename

def load_chat_history(filename):
    """Loads a chat history from a JSON file in the chat_history directory."""
    filepath = os.path.join("chat_history", filename)
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            st.session_state.messages = json.load(f)

def get_saved_chat_files():
    """Gets a list of saved chat files from the chat_history directory."""
    os.makedirs("chat_history", exist_ok=True)
    return [f for f in os.listdir("chat_history") if f.startswith("chat_") and f.endswith(".json")]

# --- Code Analysis (Placeholder) ---
def analyze_code(code: str) -> List[str]:
    """Placeholder for code analysis."""
    return ["Code analysis is not yet implemented."]

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
async def summarize_text(text: str, summary_type: str, prompt: str = "") -> str:
    """Summarizes text using the LLM, optionally based on a prompt."""
    if prompt:
        full_prompt = f"Please provide a {summary_type} of the following text, considering this prompt: '{prompt}'\n\nText: {text}"
    else:
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
        "total_tokens": 0,
        "total_cost": 0.0,
        "model_params": {"model": "gpt-3.5-turbo", "max_tokens": 150, "temperature": 0.7},
        "persona": "default",
    }
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# --- Main Application ---
def main():
    st.title("Enhanced Chat Application")
    st.sidebar.title("Settings")
    
    with st.sidebar.expander("Configuration", expanded=True):
        st.session_state.model_params["model"] = st.selectbox("Model", ["gpt-3.5-turbo", "gpt-4"], index=0)
        st.session_state.model_params["temperature"] = st.slider("Temperature", 0.0, 1.0, 0.7)
        st.session_state.model_params["max_tokens"] = st.slider("Max Tokens", 1, 2048, 150)
    
    with st.sidebar.expander("Personas", expanded=False):
        persona_options = list(PERSONAS.keys())
        selected_persona = st.selectbox("Select Persona", persona_options)
        st.session_state.persona = selected_persona
    
    with st.sidebar.expander("Upload File", expanded=False):
        uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt", "md", "jpg", "png", "py"])
        if uploaded_file:
            st.session_state.file_content = process_uploaded_file(uploaded_file)
            st.success("File processed successfully.")

    with st.sidebar.expander("Export Chat", expanded=False):
        export_format = st.selectbox("Choose export format", ["json", "md", "pdf"])
        if st.button("Export"):
            filename, data = export_chat(export_format)
            st.download_button(f"Download {export_format.upper()}", data, file_name=filename)
    
    st.chat_input("Type your message here...")

    if st.chat_input:
        user_message = st.chat_input
        validate_prompt(user_message)
        st.session_state.messages.append({"role": "user", "content": user_message})

        try:
            response = asyncio.run(create_content(user_message, st.session_state.persona))
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
            update_token_count(len(response.split()))
        except Exception as e:
            st.error(f"Error: {e}")

    if st.session_state.messages:
        for message in st.session_state.messages:
            st.chat_message(message["role"]).write(message["content"])
    
    # File operations
    saved_files = get_saved_chat_files()
    if saved_files:
        selected_file = st.selectbox("Load Previous Chat History", saved_files)
        if selected_file:
            load_chat_history(selected_file)
            st.write("Chat history loaded.")

    if st.button("Save Chat History"):
        save_chat_history()

if __name__ == "__main__":
    main()

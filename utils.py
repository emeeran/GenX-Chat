import os
import base64
from gtts import gTTS
import PyPDF2
import docx
import pytesseract
from deep_translator import GoogleTranslator
from datetime import datetime
from fpdf import FPDF
import logging
import asyncio
import aiosqlite
from PIL import Image
import streamlit as st
from dotenv import load_dotenv


# --- Global Settings and Constants ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# --- Utility Functions ---
def validate_prompt(prompt: str):
    """Validates the user prompt to ensure it's not empty."""
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty")

def process_uploaded_file(uploaded_file):
    """Processes the uploaded file based on its type."""
    file_handlers = {
        "application/pdf": lambda f: " ".join(
            page.extract_text() for page in PyPDF2.PdfReader(f).pages
        ),
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": lambda f: " ".join(
            paragraph.text for paragraph in docx.Document(f).paragraphs
        ),
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
    """Converts text to speech using gTTS."""
    lang_map = {"English": "en", "Tamil": "ta", "Hindi": "hi"}
    lang_code = lang_map.get(lang, "en")
    tts = gTTS(text=text, lang=lang_code)
    audio_file = "temp_audio.mp3"
    tts.save(audio_file)
    with open(audio_file, "rb") as f:
        audio_bytes = f.read()
    os.remove(audio_file)
    st.session_state.audio_base64 = base64.b64encode(audio_bytes).decode()

def translate_text(text: str, target_lang: str) -> str:
    """Translates text to the target language."""
    if target_lang == "English":
        return text
    translator = GoogleTranslator(source="auto", target=target_lang)
    return translator.translate(text)

def update_token_count(tokens: int):
    """Updates the token count and estimated cost in the session state."""
    st.session_state.total_tokens += tokens
    st.session_state.total_cost += tokens * 0.0001  # Assuming a cost of $0.0001 per token

def export_chat(format: str):
    """Exports the chat history in the chosen format."""
    chat_history = "\n\n".join(
        [f"**{m['role'].capitalize()}:** {m['content']}" for m in st.session_state.messages]
    )
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
async def create_database():
    """Creates the chat history database if it doesn't exist."""
    async with aiosqlite.connect("chat_history.db") as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                chat_name TEXT,
                role TEXT,
                content TEXT
            )
            """
        )
        await db.commit()

async def save_chat_history_to_db(chat_name: str):
    """Saves the current chat history to the database."""
    async with aiosqlite.connect("chat_history.db") as db:
        await db.executemany(
            "INSERT INTO chat_history (chat_name, role, content) VALUES (?, ?, ?)",
            [
                (chat_name, message["role"], message["content"])
                for message in st.session_state.messages
            ],
        )
        await db.commit()

async def load_chat_history_from_db(chat_name: str):
    """Loads chat history from the database."""
    async with aiosqlite.connect("chat_history.db") as db:
        async with db.execute(
            "SELECT role, content FROM chat_history WHERE chat_name = ? ORDER BY id",
            (chat_name,),
        ) as cursor:
            messages = [{"role": row[0], "content": row[1]} async for row in cursor]
    st.session_state.messages = messages

async def get_saved_chat_names():
    """Retrieves saved chat names from the database."""
    async with aiosqlite.connect("chat_history.db") as db:
        async with db.execute("SELECT DISTINCT chat_name FROM chat_history") as cursor:
            chat_names = [row[0] async for row in cursor]
    return chat_names

async def delete_chat(chat_name):
    """Deletes a chat by its name from the database."""
    async with aiosqlite.connect("chat_history.db") as db:
        await db.execute("DELETE FROM chat_history WHERE chat_name = ?", (chat_name,))
        await db.commit()

# --- Content Creation (Uses LLM) ---
async def create_content(prompt: str, content_type: str) -> str:
    """Uses the LLM to generate content based on the prompt and content type."""
    from main import async_stream_llm_response, get_api_client
    full_prompt = f"Write a {content_type} based on this prompt: {prompt}"
    generated_content = ""
    async for chunk in async_stream_llm_response(
        get_api_client(st.session_state.provider),
        st.session_state.model_params,
        [{"role": "user", "content": full_prompt}],
        st.session_state.provider,
        st.session_state.voice,
    ):
        generated_content += chunk
    return generated_content

# --- Summarization (Uses LLM) ---
async def summarize_text(text: str, summary_type: str) -> str:
    """Uses the LLM to summarize the given text."""
    from main import async_stream_llm_response, get_api_client
    full_prompt = f"Please provide a {summary_type} of the following text: {text}"
    summary = ""
    async for chunk in async_stream_llm_response(
        get_api_client(st.session_state.provider),
        st.session_state.model_params,
        [{"role": "user", "content": full_prompt}],
        st.session_state.provider,
        st.session_state.voice,
    ):
        summary += chunk
    return summary

# --- Helper Functions ---
def get_model_options(provider):
    """Returns a list of available model options based on the provider."""
    if provider == "Google":
        return [
            "google/gemini-1.0-pro",
            "google/gemini-1.0-pro-001",
            "google/gemini-1.5-flash-latest",
            "google/gemini-1.5-pro-latest",
        ]
    elif provider == "Groq":
        return [
            "llama-3.1-70b-versatile",
            "llama-3.1-405b-reasoning",
            "llama-3.1-8b-instant",
            "llama3-groq-70b-8192-tool-use-preview",
            "llama3-70b-8192",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
            "whisper-large-v3",
        ]
    elif provider == "OpenAI":
        return [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-3.5-turbo",
        ]
    return []

def get_max_token_limit(model):
    """Returns the maximum token limit for the specified model."""
    if "mixtral-8x7b-32768" in model:
        return 32768
    elif "llama-3.1-70b-versatile-131072" in model:
        return 131072
    elif "gemma2-9b-it" in model:
        return 8192
    return 4096

def word_count(text):
    """Returns the word count of the given text."""
    return len(text.split())

# --- Initialize Session State ---
def initialize_session_state():
    """Initializes the Streamlit session state with default values."""
    default_values = {
        "messages": [],
        "audio_base64": "",
        "file_content": "",
        "persona": "Default",
        "model_params": {
            "model": "google/gemini-1.0-pro" if GEMINI_API_KEY else "llama-3.1-70b-versatile",
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
        "provider": "Google" if GEMINI_API_KEY else "Groq" if GROQ_API_KEY else "OpenAI" if OPENAI_API_KEY else None,
        "color_scheme": "Light",
    } 
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- Reset Current Chat ---
def reset_current_chat():
    """Resets the current chat history."""
    st.session_state.messages = []
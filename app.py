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
from groq import Groq
import openai
import aiosqlite
from functools import lru_cache

# Import custom modules
from content_type import CONTENT_TYPES
from persona import PERSONAS

# --- Global Settings and Constants ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

MAX_CHAT_HISTORY_LENGTH = 50
DB_PATH = "chat_history.db"

# --- Voice Options ---
VOICE_OPTIONS = {
    "OpenAI": [
        "alloy",
        "echo",
        "fable",
        "onyx",
        "nova",
        "shimmer",
    ],
    "gTTS": ["en", "ta", "hi"],
}

# --- Gemini Function Declarations ---
GEMINI_FUNCTIONS = [
    {
        "name": "get_weather",
        "description": "Get weather information for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The name of the city or location.",
                },
                "date": {
                    "type": "string",
                    "description": "The date for which to retrieve the weather (format: YYYY-MM-DD).",
                    "optional": True,
                },
            },
            "required": ["location"],
        },
    },
    {
        "name": "compute_sum",
        "description": "Computes the sum of two numbers.",
        "parameters": {
            "type": "object",
            "properties": {
                "number1": {"type": "number", "description": "The first number."},
                "number2": {"type": "number", "description": "The second number."},
            },
            "required": ["number1", "number2"],
        },
    },
    {
        "name": "translate_text",
        "description": "Translates text from one language to another.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text to translate."},
                "target_language": {
                    "type": "string",
                    "description": "The language code to translate the text into.",
                },
            },
            "required": ["text", "target_language"],
        },
    },
    {
        "name": "generate_story",
        "description": "Generates a short story based on a given theme.",
        "parameters": {
            "type": "object",
            "properties": {
                "theme": {
                    "type": "string",
                    "description": "The theme for the short story.",
                },
                "length": {
                    "type": "integer",
                    "description": "The desired length of the story in paragraphs.",
                    "optional": True,
                },
            },
            "required": ["theme"],
        },
    },
]


# --- Utility Functions ---
@lru_cache(maxsize=None)
def get_api_client(provider: str):
    """Returns the appropriate API client based on the selected provider."""
    try:
        if provider == "Groq":
            return Groq(api_key=GROQ_API_KEY)
        elif provider == "OpenAI":
            return openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
        elif provider == "Google":
            return None  # Gemini doesn't use a client object
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    except Exception as e:
        logger.error(f"Failed to initialize {provider} client: {str(e)}")
        return None


async def create_database():
    """Creates the chat history database if it doesn't exist."""
    async with aiosqlite.connect(DB_PATH) as db:
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


async def async_stream_llm_response(
    client,
    params: Dict[str, Any],
    messages: List[Dict[str, str]],
    provider: str,
    voice: str = "alloy",
):
    """Streams the LLM response asynchronously."""
    try:
        if provider == "Groq":
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {GROQ_API_KEY}",
                        "Content-Type": "application/json",
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
                        logger.error(
                            f"Groq API Error: {response.status} - {error_text}"
                        )
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
                                logger.error(
                                    f"JSON decode error: {json_err}. Raw line: {line}"
                                )
                            except Exception as e:
                                logger.error(
                                    f"Error processing line: {e}. Raw line: {line}"
                                )
        elif provider == "OpenAI":
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

        elif provider == "Google":
            request_body = {
                "model": params["model"],
                "prompt": {
                    "text": messages[-1]["content"],
                },
                "tool_code": {"function_declarations": GEMINI_FUNCTIONS},
            }

            headers = {
                "Authorization": f"Bearer {GEMINI_API_KEY}",
                "Content-Type": "application/json",
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.google.com/v1/generative/text:generateText",
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
                                function_args = json.loads(
                                    function_call.get("arguments", "{}")
                                )
                                yield f"Function Call: {function_name}\nArguments: {function_args}"
                            else:
                                yield candidate["output"]
                        else:
                            logger.error(f"Google API response error: {data}")
                            yield f"Error in Google API response: {data}"
                    else:
                        error_text = await response.text()
                        logger.error(
                            f"Google API Error: {response.status} - {error_text}"
                        )
                        yield f"Google API Error: {response.status} - {error_text}"

    except Exception as e:
        logger.error(f"Error in API call: {str(e)}")
        yield f"Error in API call: {str(e)}"


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
    st.session_state.total_cost += (
        tokens * 0.0001
    )  # Assuming a cost of $0.0001 per token


def export_chat(format: str):
    """Exports the chat history in the chosen format."""
    chat_history = "\n\n".join(
        [
            f"**{m['role'].capitalize()}:** {m['content']}"
            for m in st.session_state.messages
        ]
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
async def save_chat_history_to_db(chat_name: str):
    """Saves the current chat history to the database."""
    async with aiosqlite.connect(DB_PATH) as db:
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
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT role, content FROM chat_history WHERE chat_name = ? ORDER BY id",
            (chat_name,),
        ) as cursor:
            messages = [{"role": row[0], "content": row[1]} async for row in cursor]
    st.session_state.messages = messages


async def get_saved_chat_names():
    """Retrieves saved chat names from the database."""
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT DISTINCT chat_name FROM chat_history") as cursor:
            chat_names = [row[0] async for row in cursor]
    return chat_names


async def delete_chat(chat_name):
    """Deletes a chat by its name from the database."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM chat_history WHERE chat_name = ?", (chat_name,))
        await db.commit()


# --- Content Creation (Uses LLM) ---
async def create_content(prompt: str, content_type: str) -> str:
    """Uses the LLM to generate content based on the prompt and content type."""
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
            "model": (
                "google/gemini-1.0-pro" if GEMINI_API_KEY else "llama-3.1-70b-versatile"
            ),
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
        "provider": (
            "Google"
            if GEMINI_API_KEY
            else "Groq" if GROQ_API_KEY else "OpenAI" if OPENAI_API_KEY else None
        ),
        "color_scheme": "Light",
    }
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value


# --- Reset Current Chat ---
def reset_current_chat():
    """Resets the current chat history."""
    st.session_state.messages = []


# --- Main Streamlit Application ---
async def main():
    """Main function for the Streamlit application."""
    initialize_session_state()

    # Set page configuration at the top of the main function
    st.set_page_config(
        page_title="GenX-Chat",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    with st.spinner("Creating database..."):
        await create_database()

    # Select API provider
    # Center align the title
    st.sidebar.markdown("<h3 style='text-align: center;'>Select Provider</h3>", unsafe_allow_html=True)
    provider_options = ["Google", "Groq", "OpenAI"]

    # Ensure the available providers are based on the keys provided
    available_providers = [
        p
        for p in provider_options
        if (
            (p == "Google" and GEMINI_API_KEY)
            or (p == "Groq" and GROQ_API_KEY)
            or (p == "OpenAI" and OPENAI_API_KEY)
        )
    ]

    if not available_providers:
        st.error(
            "No API keys are set. Please set at least one API key in your .env file."
        )
        st.stop()

    selected_provider = st.sidebar.selectbox(
        "Select Provider",
        available_providers,
        index=(
            available_providers.index(st.session_state.provider)
            if st.session_state.provider in available_providers
            else 0
        ),
    )

    # Initialize the client based on the selected provider
    if selected_provider == "Google":
        client = get_api_client("Google")
        st.session_state.provider = "Google"
    else:
        client = get_api_client(selected_provider)
        st.session_state.provider = selected_provider

    if client is None and st.session_state.provider != "Google":
        st.error(
            f"Failed to initialize {selected_provider} client. Please check your API key and try again."
        )
        return

    # --- Sidebar ---
    with st.sidebar:
        st.markdown(
            "<h3 style='text-align: center;'>GenX-Chat Settings</h3>",
            unsafe_allow_html=True,
        )

        # Chat Settings
        with st.expander("Chat Settings", expanded=False):
            saved_chats = await get_saved_chat_names()
            selected_chat = st.selectbox(
                "Load Chat History", options=[""] + saved_chats
            )
            if selected_chat:
                await load_chat_history_from_db(selected_chat)

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Save Chat"):
                    chat_name = st.text_input("Enter a name for this chat:")
                    if chat_name:
                        await save_chat_history_to_db(chat_name)
                        st.success(f"Chat '{chat_name}' saved successfully.")
            with col2:
                reset_button = st.button("Reset Chat", on_click=reset_current_chat)
            with col3:
                if st.button("Delete Chat"):
                    if selected_chat:
                        await delete_chat(selected_chat)
                        st.success(f"Chat '{selected_chat}' deleted successfully.")
                        st.experimental_rerun()

        # Model Settings
        with st.expander("Model"):
            model_options = get_model_options(st.session_state.provider)
            st.session_state.model_params["model"] = st.selectbox(
                "Choose Model:", options=model_options
            )

            max_token_limit = get_max_token_limit(
                st.session_state.model_params["model"]
            )
            st.session_state.model_params["max_tokens"] = st.slider(
                "Max Tokens:",
                min_value=1,
                max_value=max_token_limit,
                value=min(1024, max_token_limit),
                step=1,
            )
            st.session_state.model_params["temperature"] = st.slider(
                "Temperature:", 0.0, 2.0, 1.0, 0.1
            )
            st.session_state.model_params["top_p"] = st.slider(
                "Top-p:", 0.0, 1.0, 1.0, 0.1
            )

        # Persona Settings
        with st.expander("Persona"):
            persona_options = list(PERSONAS.keys())
            st.session_state.persona = st.selectbox(
                "Select Persona:",
                options=persona_options,
                index=persona_options.index("Default"),
            )
            st.text_area(
                "Persona Description:",
                value=PERSONAS[st.session_state.persona],
                height=100,
                disabled=True,
            )

        # Audio & Language Settings
        with st.expander("Audio & Language"):
            st.session_state.enable_audio = st.checkbox(
                "Enable Audio Response", value=False
            )
            language_options = ["English", "Tamil", "Hindi"]
            st.session_state.language = st.selectbox(
                "Select Language:", language_options
            )
            if st.session_state.provider == "OpenAI":
                st.session_state.voice = st.selectbox(
                    "Select Voice (OpenAI):", VOICE_OPTIONS["OpenAI"]
                )
            else:
                st.session_state.voice = st.selectbox(
                    "Select Language Code (gTTS):", VOICE_OPTIONS["gTTS"]
                )

        # File Upload
        with st.expander("File Upload"):
            uploaded_file = st.file_uploader(
                "Upload a file", type=["pdf", "docx", "txt", "md", "jpg", "jpeg", "png"]
            )
            if uploaded_file:
                try:
                    st.session_state.file_content = process_uploaded_file(uploaded_file)
                    st.success("File processed successfully")
                except Exception as e:
                    st.error(f"Error processing file: {e}")

        # Summarization
        with st.expander("Summarize"):
            st.session_state.show_summarization = st.checkbox(
                "Enable Summarization", value=False
            )
            if st.session_state.show_summarization:
                st.session_state.summarization_type = st.selectbox(
                    "Summarization Type:",
                    [
                        "Main Takeaways",
                        "Main points bulleted",
                        "Concise Summary",
                        "Executive Summary",
                    ],
                )

        # Content Generation
        with st.expander("Content Generation"):
            st.session_state.content_creation_mode = st.checkbox(
                "Enable Content Creation Mode", value=False
            )
            if st.session_state.content_creation_mode:
                st.session_state.content_type = st.selectbox(
                    "Select Content Type:", list(CONTENT_TYPES.keys())
                )

        # Export Chat
        with st.expander("Export"):
            export_format = st.selectbox("Export Format", ["md", "pdf"])
            st.button("Export Chat", on_click=lambda: export_chat(export_format))

        # Color Scheme
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

    # --- Main Chat Interface ---
    st.markdown(
        '<h1 style="text-align: center; color: #6ca395;">GenX-Chat 💬</h1>',
        unsafe_allow_html=True,
    )

    # Chat interface
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages[-MAX_CHAT_HISTORY_LENGTH:]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Input
    prompt = st.chat_input("Enter your message:")
    if prompt:
        await process_chat_input(prompt, client)

    # Display metrics
    total_words = sum(word_count(msg["content"]) for msg in st.session_state.messages)
    st.sidebar.metric("Total Words", total_words)
    st.sidebar.metric("Total Tokens", st.session_state.total_tokens)
    st.sidebar.metric("Estimated Cost", f"${st.session_state.total_cost:.4f}")


async def process_chat_input(prompt: str, client):
    """Processes the user's chat input."""
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
                if chunk.startswith("API Error:") or chunk.startswith(
                    "Error in API call:"
                ):
                    message_placeholder.error(chunk)
                    return
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )

        if st.session_state.enable_audio and full_response.strip():
            if st.session_state.provider == "Groq":
                text_to_speech(full_response, st.session_state.language)
            st.audio(
                f"data:audio/mp3;base64,{st.session_state.audio_base64}",
                format="audio/mp3",
            )

        update_token_count(len(full_response.split()))

        # Handle content creation
        if st.session_state.content_creation_mode:
            content_type = CONTENT_TYPES[st.session_state.content_type]
            generated_content = await create_content(prompt, content_type)
            with st.chat_message("assistant"):
                st.markdown(
                    f"## Generated {st.session_state.content_type}:\n\n{generated_content}"
                )

        # Handle summarization
        if st.session_state.show_summarization:
            text_to_summarize = (
                st.session_state.file_content
                if st.session_state.file_content
                else prompt
            )
            summary = await summarize_text(
                text_to_summarize, st.session_state.summarization_type
            )
            with st.chat_message("assistant"):
                st.markdown(
                    f"## Summary ({st.session_state.summarization_type}):\n\n{summary}"
                )

    except ValueError as ve:
        st.error(f"Invalid input: {str(ve)}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        logger.error(f"Unexpected error in process_chat_input: {str(e)}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
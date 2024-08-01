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
from streamlit_ace import st_ace
import streamlit as st
import html


from persona import PERSONAS

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
if API_KEY is None:
    st.error("GROQ_API_KEY environment variable not set. Please set it in your .env file.")
    st.stop()

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

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
        "text/x-python": lambda f: f.getvalue().decode("utf-8"),
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
        data = st.session_state.messages
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
        return filename, data 

    elif format == "md":
        filename = f"chat_history_{timestamp}.md"
        with open(filename, "w") as f:
            f.write(chat_history_text)
        return filename, chat_history_text

    elif format == "pdf":
        filename = f"chat_history_{timestamp}.pdf"
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, chat_history_text)
        pdf.output(filename)
        return filename, None  

# --- Chat History Management ---
def save_chat_history():
    """Saves the chat history to a JSON file."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"./chat_history/chat_{timestamp}.json"
    os.makedirs("./chat_history", exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(st.session_state.messages, f, indent=4)

def load_chat_history(filename):
    """Loads a chat history from a JSON file."""
    filepath = os.path.join("./chat_history", filename)
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            st.session_state.messages = json.load(f)

def get_saved_chat_files():
    """Gets a list of saved chat files."""
    return [f for f in os.listdir("./chat_history") if f.endswith(".json")]

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
        "chat_histories": [],
        "persona": "Default",
        "custom_persona": "",
        "model_params": {"model": "llama-3.1-70b-versatile", "max_tokens": 11024, "temperature": 1.0, "top_p": 1.0},
        "total_tokens": 0,
        "total_cost": 0,
        "enable_audio": False,
        "language": "English",
        "show_analysis": False,
        "content_creation_mode": False,
        "content_type": "Story",
        "show_summarization": False,
        "summarization_type": "Main Takeaways",
        "text_to_summarize": "",
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
        page_title="GenX Chat",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown('<h1 style="text-align: center; color: #6ca395;">GenX Chat 💬</h1>', unsafe_allow_html=True)

    # --- Sidebar ---
    with st.sidebar:

        st.markdown(
            '<h1 style="text-align: center; color: #6ca395;"> Welcome to GenX Chat!</h1>',
            unsafe_allow_html=True,
        )
        st.write(
            "GenX Chat is a fast multimodal chat app that uses the Groq API to generate responses to your prompts. You can chat with the AI, upload files, generate content, summarize text, and more!"
        )

        st.title("🔧 Settings")

        # Chat Settings (Expanded by default)
        with st.expander("Chat Settings", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.button(
                    "Reset Chat",
                    on_click=lambda: st.session_state.update(
                        {
                            "messages": [],
                            "file_content": "",
                            "total_tokens": 0,
                            "total_cost": 0,
                        }
                    ),
                )
            with col2:
                st.button("Save Chat", on_click=save_chat_history)

            saved_chats = get_saved_chat_files()
            selected_chat = st.selectbox("Load Chat History", options=saved_chats)
            if selected_chat:
                load_chat_history(selected_chat)

        # Model Settings
        with st.expander("Model Settings"):
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
                    "whisper-large-v3",
                ],
            )

            # Adjust max tokens based on model
            max_token_limit = 4096
            if st.session_state.model_params["model"] == "mixtral-8x7b-32768":
                max_token_limit = 32768
            elif (
                st.session_state.model_params["model"]
                == "llama-3.1-70b-versatile-131072"
            ):
                max_token_limit = 131072
            elif st.session_state.model_params["model"] == "gemma2-9b-it":
                max_token_limit = 8192

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
            st.session_state.model_params["top_p"] = st.slider("Top-p:", 0.0, 1.0, 1.0, 0.1)

        # Persona Settings
        with st.expander("Persona Settings"):
            persona_options = list(PERSONAS.keys()) + ["Custom"]
            st.session_state.persona = st.selectbox(
                "Select Persona:",
                options=persona_options,
                index=persona_options.index("Default"),
            )

            if st.session_state.persona == "Custom":
                st.session_state.custom_persona = st.text_area(
                    "Custom Persona Description:",
                    value=st.session_state.custom_persona,
                    height=100,
                )
            else:
                st.text_area(
                    "Persona Description:",
                    value=PERSONAS.get(st.session_state.persona, ""),
                    height=100,
                    disabled=True,
                )

        # Audio & Language Settings
        with st.expander("Audio & Language"):
            st.session_state.enable_audio = st.checkbox(
                "Enable Audio Response", value=False
            )
            st.session_state.language = st.selectbox(
                "Select Language:", ["English", "Tamil", "Hindi"]
            )

        # Content Creation Mode
        with st.expander("Content Creation"):
            st.session_state.content_creation_mode = st.checkbox(
                "Enable Content Creation Mode", value=False
            )
            if st.session_state.content_creation_mode:
                st.session_state.content_type = st.selectbox(
                    "Select Content Type:", ["Story", "Poem", "Article"]
                )

        # Summarization Mode
        with st.expander("Summarization"):
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

        # Export Chat
        with st.expander("Export Chat"):
            export_format = st.selectbox(
                "Select Export Format", ["json", "md", "pdf"]
            )
            filename, data = export_chat(export_format)
            if data:
                st.download_button(
                    f"Export Chat as {export_format.upper()}",
                    data=data,
                    file_name=filename,
                )
            else:
                st.download_button(
                    f"Export Chat as {export_format.upper()}",
                    data=open(filename, "rb"),
                    file_name=filename,
                )

        # File Upload
        with st.expander("File Upload"):
            uploaded_file = st.file_uploader(
                "Upload a file",
                type=["pdf", "docx", "txt", "md", "jpg", "jpeg", "png", "py"],
            )
            if uploaded_file:
                try:
                    st.session_state.file_content = process_uploaded_file(
                        uploaded_file
                    )
                    st.success("File processed successfully")
                    if uploaded_file.type == "text/x-python":
                        st.session_state.show_analysis = st.checkbox(
                            "Show Code Analysis", value=False
                        )
                        if st.session_state.show_analysis:
                            analysis_result = analyze_code(
                                st.session_state.file_content
                            )
                            for result in analysis_result:
                                st.write(result)
                except Exception as e:
                    st.error(f"Error processing file: {e}")

        st.markdown(
            "Created & Maintained: <b> Meeran E Mandhini <b> emeeranjp@gmail.com",
            unsafe_allow_html=True,
        )


# --- Main Chat Interface ---
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # --- Prompt Input ---
    prompt = st.chat_input("Enter your message:")
    if prompt:
        asyncio.run(process_chat_input(prompt, client))

    st.markdown(
        '<script>window.scrollTo(0,document.body.scrollHeight);</script>',
        unsafe_allow_html=True,
    )

async def process_chat_input(prompt: str, client: Groq):
    """Processes chat input, handling summarization or regular chat."""
    try:
        validate_prompt(prompt)

        if st.session_state.show_summarization:
            text_to_summarize = st.session_state.file_content 
            if text_to_summarize:
                with st.spinner("Summarizing..."):
                    summary = await summarize_text(
                        text_to_summarize,
                        st.session_state.summarization_type,
                        prompt,
                    )
                    # Add the prompt and summary to the chat history:
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    st.session_state.messages.append({"role": "assistant", "content": summary})
            else:
                st.warning("Please upload a file or paste text to summarize.")
        
        else:            # Regular Chat Mode
            if st.session_state.file_content:
                prompt = f"Based on the uploaded file content, {prompt}\n\nFile content: {st.session_state.file_content[:4000]}..."

            if (
                st.session_state.persona == "Custom"
                and st.session_state.custom_persona
            ):
                system_message = st.session_state.custom_persona
            else:
                system_message = PERSONAS.get(st.session_state.persona, "")

            messages = [
                {"role": "system", "content": system_message},
                *st.session_state.messages,
                {"role": "user", "content": prompt},
            ]

            full_response = ""
            message_placeholder = st.empty()
            async for chunk in async_stream_llm_response(
                client, st.session_state.model_params, messages
            ):
                if chunk.startswith("API Error:") or chunk.startswith(
                    "Error in API call:"
                ):
                    message_placeholder.error(chunk)
                    return
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")

            if not full_response:
                message_placeholder.error(
                    "No response generated. Please try again."
                )
                return

            message_placeholder.markdown(full_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )

            if st.session_state.enable_audio and full_response.strip():
                text_to_speech(full_response, st.session_state.language)
                st.audio(
                    f"data:audio/mp3;base64,{st.session_state.audio_base64}",
                    format="audio/mp3",
                )

            update_token_count(len(full_response.split()))
            
    except Exception as e:
        st.error(f"Error processing chat input: {str(e)}")


if __name__ == "__main__":
    if not API_KEY:
        st.error("GROQ_API_KEY is not set. Please check your .env file.")
        st.stop()
    main()
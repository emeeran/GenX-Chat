import streamlit as st
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
from fpdf import FPDF
from dotenv import load_dotenv
import groq
import ast
import logging
from typing import List, Dict, Any
import asyncio
import aiohttp
from streamlit_ace import st_ace
import html
import json

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load persona definitions
PERSONAS = {
    "Default": "You are a helpful assistant.",
    "Programmer": "You are a skilled programmer with expertise in multiple languages.",
    "Creative Writer": "You are a creative writer with a flair for storytelling.",
    "Data Analyst": "You are a data analyst with expertise in statistics and data visualization.",
}

# Set page config
st.set_page_config(
    page_title="Enhanced Groq-Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
def initialize_session_state():
    default_values = {
        "messages": [],
        "audio_base64": "",
        "file_content": "",
        "chat_histories": [],
        "persona": "Default",
        "user": None,
        "model_params": {
            "model": "llama-3.1-70b-versatile",
            "max_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.9,
        },
        "total_tokens": 0,
        "total_cost": 0,
        "enable_audio": False,
        "language": "English",
        "show_analysis": False,
        "content_creation_mode": False,
        "show_summarization": False,
        "summarization_type": "Main Takeaways",
        "code_editor_content": "",
    }
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# Groq client setup
def get_groq_client(api_key: str):
    return groq.Groq(api_key=api_key)

async def async_stream_llm_response(client, params: Dict[str, Any], messages: List[Dict[str, str]]):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.groq.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": params["model"],
                    "messages": messages,
                    "max_tokens": params["max_tokens"],
                    "temperature": params["temperature"],
                    "top_p": params["top_p"],
                    "stream": True,
                },
            ) as response:
                async for line in response.content:
                    if line.startswith(b"data: "):
                        chunk = json.loads(line[6:])
                        if chunk["choices"][0]["delta"].get("content"):
                            yield chunk["choices"][0]["delta"]["content"]
    except Exception as e:
        logger.error(f"Error in API call: {str(e)}")
        yield "I'm sorry, but I encountered an error while processing your request."

def validate_prompt(prompt: str):
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty")

def analyze_code(code: str) -> List[Dict[str, Any]]:
    try:
        tree = ast.parse(code)
        analysis_results = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_name = node.name
                docstring = ast.get_docstring(node)
                arguments = [arg.arg for arg in node.args.args]
                analysis_results.append(
                    {
                        "Function": function_name,
                        "Docstring": docstring,
                        "Arguments": arguments,
                    }
                )
        return analysis_results
    except SyntaxError as e:
        return [{"Error": f"Syntax Error: {e.msg}"}]

async def summarize_text(text: str, summarization_type: str = "Main Takeaways") -> str:
    prompt_templates = {
        "Main Takeaways": "Please provide the main takeaways of the following text:\n\n{text}",
        "Main points bulleted": "Please list the main points of the following text in bullet points:\n\n{text}",
        "Concise Summary": "Please provide a concise summary of the following text:\n\n{text}",
        "Executive Summary": "Please provide an executive summary of the following text:\n\n{text}",
    }
    prompt = prompt_templates.get(summarization_type, "Please summarize the following text:\n\n{text}")
    prompt = prompt.format(text=text)

    messages = [{"role": "user", "content": prompt}]
    client = get_groq_client(API_KEY)
    response = await client.chat.completions.create(
        messages=messages,
        model=st.session_state.model_params["model"],
        max_tokens=150,
        temperature=0.7,
        top_p=0.9,
    )
    return response.choices[0].message.content

async def create_content(prompt: str, content_type: str = "Story") -> str:
    messages = [
        {
            "role": "system",
            "content": f"You are a creative writer. Generate a {content_type.lower()} based on the user's prompt.",
        },
        {"role": "user", "content": prompt},
    ]
    client = get_groq_client(API_KEY)
    response = await client.chat.completions.create(
        messages=messages,
        model=st.session_state.model_params["model"],
        max_tokens=500,
        temperature=0.8,
        top_p=0.9,
    )
    return response.choices[0].message.content

def export_chat(format: str):
    chat_history = "\n\n".join(
        [
            f"**{m['role'].capitalize()}:** {m['content']}"
            for m in st.session_state.messages
        ]
    )
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"chat_exports/chat_history_{timestamp}.{format}"
    os.makedirs("chat_exports", exist_ok=True)

    with open(filename, "w" if format == "md" else "wb") as f:
        if format == "md":
            f.write(chat_history)
        elif format == "pdf":
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, chat_history)
            pdf.output(f)

    with open(filename, "rb") as f:
        st.download_button(f"Download {format.upper()}", f, file_name=filename)

def process_uploaded_file(uploaded_file):
    file_type_handlers = {
        "application/pdf": lambda f: " ".join(
            page.extract_text() for page in PyPDF2.PdfReader(f).pages
        ),
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": lambda f: " ".join(
            paragraph.text for paragraph in docx.Document(f).paragraphs
        ),
        "text/plain": lambda f: f.getvalue().decode("utf-8"),
        "text/markdown": lambda f: f.getvalue().decode("utf-8"),
        "image/": lambda f: pytesseract.image_to_string(Image.open(f)),
        "text/x-python": lambda f: f.getvalue().decode("utf-8"),
    }
    for file_type, handler in file_type_handlers.items():
        if uploaded_file.type.startswith(file_type):
            return handler(uploaded_file)
    raise ValueError("Unsupported file type")

def save_chat_history():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    st.session_state.chat_histories.append(
        {
            "name": f"Chat {timestamp}",
            "messages": st.session_state.messages.copy(),
        }
    )

def load_chat_history(selected_history: str):
    for history in st.session_state.chat_histories:
        if history["name"] == selected_history:
            st.session_state.messages = history["messages"].copy()
            break

def update_token_count(tokens: int):
    st.session_state.total_tokens += tokens
    st.session_state.total_cost += tokens * 0.0001

def text_to_speech(text: str, lang: str):
    if not text.strip():
        st.warning("No text to speak")
        return
    
    lang_map = {"english": "en", "tamil": "ta", "hindi": "hi"}
    lang_code = lang_map.get(lang.lower(), "en")
    tts = gTTS(text=text, lang=lang_code)
    audio_file = "temp_audio.mp3"
    tts.save(audio_file)
    with open(audio_file, "rb") as f:
        audio_bytes = f.read()
    os.remove(audio_file)
    st.session_state.audio_base64 = base64.b64encode(audio_bytes).decode()

def reset_all():
    st.session_state.update(
        {
            "messages": [],
            "total_tokens": 0,
            "total_cost": 0,
            "file_content": "",
            "audio_base64": "",
            "code_editor_content": "",
        }
    )

def translate_text(text: str, target_lang: str) -> str:
    if target_lang == "English":
        return text
    translator = GoogleTranslator(source="auto", target=target_lang)
    return translator.translate(text)

async def process_chat_input(prompt: str):
    if st.session_state.model_params["model"] == "whisper-large-v3":
        st.error("Whisper model is for speech recognition only. Please select a different model for text chat.")
        return

    try:
        validate_prompt(prompt)

        if st.session_state.file_content:
            prompt = f"Based on the uploaded file content, {prompt}\n\nFile content: {st.session_state.file_content[:4000]}..."

        messages = [
            {"role": "system", "content": PERSONAS[st.session_state.persona]},
            *st.session_state.messages,
            {"role": "user", "content": prompt},
        ]

        full_response = ""
        message_placeholder = st.chat_message("assistant") # Placeholder for assistant's response
        async for chunk in async_stream_llm_response(
            get_groq_client(API_KEY),
            st.session_state.model_params,
            messages,
        ):
            full_response += chunk
            message_placeholder.markdown(full_response + "▌") # Update the placeholder with the streaming chunk

        message_placeholder.markdown(full_response) # Update with the complete response

        st.session_state.messages.append({"role": "assistant", "content": full_response})

        if st.session_state.enable_audio and full_response.strip():
            text_to_speech(full_response, st.session_state.language)

        update_token_count(len(full_response.split()))
    except ValueError as e:
        st.error(str(e))
def main():
    st.markdown(
        """<h1 style="text-align: center; color: #6ca395;">Enhanced Groq-Chat 💬</h1>""",
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.title("🔧 Settings")

        # Chat Settings
        with st.expander("Chat Settings"):
            col1, col2 = st.columns(2)
            with col1:
                st.button("Reset All", on_click=reset_all)
            with col2:
                st.button("Save Chat History", on_click=save_chat_history)

            chat_history_names = [history["name"] for history in st.session_state.chat_histories]
            selected_history = st.selectbox("Load Chat History", options=[""] + chat_history_names)
            if selected_history:
                st.button("Load Chat History", on_click=load_chat_history, args=(selected_history,))

        # Audio Settings
        with st.expander("Audio Settings"):
            st.session_state.enable_audio = st.checkbox("Enable Audio Response", value=False)
            st.session_state.language = st.selectbox("Select Language:", ["English", "Tamil", "Hindi"])

        # Persona Settings
        with st.expander("Persona Settings"):
            persona_options = list(PERSONAS.keys())
            st.session_state.persona = st.selectbox("Select Persona:", options=persona_options, index=persona_options.index("Default"))
            st.text_area("Persona Description:", value=PERSONAS[st.session_state.persona], height=100, disabled=True)

        # Model Settings
        with st.expander("Model Settings"):
            st.session_state.model_params["model"] = st.selectbox(
                "Choose Model:",
                options=[
                    "llama-3.1-405b-reasoning",
                    "llama-3.1-70b-versatile",
                    "llama-3.1-8b-instant",
                    "llama3-groq-70b-8192-tool-use-preview",
                    "llama3-70b-8192",
                    "mixtral-8x7b-32768",
                    "gemma2-9b-it",
                    "whisper-large-v3"
                ],
            )
            
            # Adjust max tokens based on model
            if st.session_state.model_params["model"] in ["llama3-groq-70b-8192-tool-use-preview", "llama3-70b-8192"]:
                max_token_limit = 8192
            elif st.session_state.model_params["model"] == "mixtral-8x7b-32768":
                max_token_limit = 32768
            else:
                max_token_limit = 4096  # Default for other models
            
            st.session_state.model_params["max_tokens"] = st.slider("Max Tokens:", min_value=1, max_value=max_token_limit, value=min(1024, max_token_limit), step=1)
            
            # Only show temperature and top_p for non-whisper models
            if st.session_state.model_params["model"] != "whisper-large-v3":
                st.session_state.model_params["temperature"] = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
                st.session_state.model_params["top_p"] = st.slider("Top-p:", min_value=0.0, max_value=1.0, value=0.9, step=0.1)
            
            # Special handling for whisper model
            if st.session_state.model_params["model"] == "whisper-large-v3":
                st.warning("Whisper model is designed for speech recognition. Please upload an audio file for transcription.")
                audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])
                if audio_file is not None:
                    # Here you would implement the logic to transcribe the audio file using the whisper model
                    st.info("Audio transcription feature is not yet implemented.")

        # File Upload and Code Analysis
        with st.expander("File Upload"):
            uploaded_file = st.file_uploader("Upload a file", type=["pdf", "docx", "txt", "md", "jpg", "jpeg", "png", "py"])
            if uploaded_file is not None:
                try:
                    st.session_state.file_content = process_uploaded_file(uploaded_file)
                    st.success("File processed successfully")
                    if uploaded_file.type == "text/x-python":
                        st.session_state.show_analysis = st.checkbox("Show Code Analysis", value=False)
                        if st.session_state.show_analysis:
                            analysis_result = analyze_code(st.session_state.file_content)
                            for result in analysis_result:
                                st.write(result)
                except Exception as e:
                    st.error(f"Error processing file: {e}")

        # Export Chat
        with st.expander("Export Chat"):
            export_format = st.selectbox("Export Format", ["md", "pdf"])
            if st.button("Export Chat"):
                export_chat(export_format)

        # Content Creation Mode
        with st.expander("Content Creation"):
            st.session_state.content_creation_mode = st.checkbox("Enable Content Creation Mode", value=False)
            if st.session_state.content_creation_mode:
                content_prompt = st.text_area("Enter your creative prompt:")
                content_type = st.selectbox("Select Content Type:", ["Story", "Poem", "Article"])
                if st.button("Generate Content"):
                    if content_prompt:
                        with st.spinner("Generating..."):
                            generated_content = asyncio.run(create_content(content_prompt, content_type))
                            st.write("## Generated Content:")
                            st.write(generated_content)
                    else:
                        st.warning("Please enter a prompt.")

        # Summarization Mode
        with st.expander("Summarization"):
            st.session_state.show_summarization = st.checkbox("Enable Summarization", value=False)
            if st.session_state.show_summarization:
                st.session_state.summarization_type = st.selectbox(
                    "Summarization Type:",
                    ["Main Takeaways", "Main points bulleted", "Concise Summary", "Executive Summary"],
                )
                if st.button("Summarize"):
                    text_to_summarize = st.session_state.file_content if st.session_state.file_content else st.session_state.get("text_to_summarize", "")
                    if text_to_summarize:
                        with st.spinner("Summarizing..."):
                            summary = asyncio.run(summarize_text(text_to_summarize, st.session_state.summarization_type))
                            st.write("## Summary:")
                            st.write(summary)
                    else:
                        st.warning("Please upload a file or enter text to summarize.")

        # Code Editor
        with st.expander("Code Editor"):
            st.session_state.code_editor_content = st_ace(
                value=st.session_state.code_editor_content,
                language="python",
                theme="monokai",
                key="code_editor"
            )
            if st.button("Run Code"):
                try:
                    exec(st.session_state.code_editor_content)
                except Exception as e:
                    st.error(f"Error executing code: {str(e)}")

    # Main Chat Interface
    st.write("## Chat")
    
    # Display existing messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Prompt Input at the Bottom
    prompt = st.chat_input("Enter your message:")
    if prompt:
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Process the chat input
        asyncio.run(process_chat_input(prompt))
        
    # Audio Output
    if st.session_state.audio_base64:
        audio_bytes = base64.b64decode(st.session_state.audio_base64)
        st.audio(audio_bytes, format="audio/mp3")

    # Statistics
    st.sidebar.markdown("### Statistics")
    st.sidebar.write(f"Total Tokens Used: {st.session_state.total_tokens}")
    st.sidebar.write(f"Total Cost (USD): ${st.session_state.total_cost:.4f}")


if __name__ == "__main__":
    main()
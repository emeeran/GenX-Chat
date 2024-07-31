import streamlit as st
import os
import sys
import base64
from gtts import gTTS
import PyPDF2
import docx  # Correct import for handling DOCX files
import markdown
from PIL import Image
import pytesseract
import io
from deep_translator import GoogleTranslator
from datetime import datetime
from fpdf import FPDF
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()




# Set page config at the very beginning
st.set_page_config(page_title="Groq-Chat", page_icon="", layout="wide", initial_sidebar_state="expanded")

# Function to load configuration
def load_config():
    return {
        'api_key': os.getenv('GROQ_API_KEY'),
        'models': {
            'llama-3.1-405b-reasoning': {
                'name': 'Llama 3.1 405B (Preview)',
                'tokens': 16000
            },
            'llama-3.1-70b-versatile': {
                'name': 'Llama 3.1 70B (Preview)',
                'tokens': 8000
            },
            'llama-3.1-8b-instant': {
                'name': 'Llama 3.1 8B (Preview)',
                'tokens': 8000
            },
            'llama3-groq-70b-8192-tool-use-preview': {
                'name': 'Llama 3 Groq 70B Tool Use (Preview)',
                'tokens': 8192
            },
            'llama3-groq-8b-8192-tool-use-preview': {
                'name': 'Llama 3 Groq 8B Tool Use (Preview)',
                'tokens': 8192
            },
            'llama3-70b-8192': {
                'name': 'Meta Llama 3 70B',
                'tokens': 8192
            },
            'llama3-8b-8192': {
                'name': 'Meta Llama 3 8B',
                'tokens': 8192
            },
            'mixtral-8x7b-32768': {
                'name': 'Mixtral 8x7B',
                'tokens': 32768
            },
            'gemma-7b-it': {
                'name': 'Gemma 7B',
                'tokens': 8192
            },
            'gemma2-9b-it': {
                'name': 'Gemma 2 9B',
                'tokens': 8192
            },
            'whisper-large-v3': {
                'name': 'Whisper',
                'tokens': 8192
            },
        }
    }

# Function to get Groq client
def get_groq_client(api_key):
    return api_key

# Function to stream LLM response
def stream_llm_response(api_key, model_params, messages):
    # Simulated response
    yield "Simulated response. Replace with actual API call."

# Function to validate prompt
def validate_prompt(prompt):
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty")

# Initialize session state
def initialize_session_state():
    default_values = {
        "messages": [],
        "audio_base64": "",
        "file_content": "",
        "chat_histories": [],
        "persona": "Default Persona",
        "user": None,
        "model_params": {},
        "total_tokens": 0,
        "total_cost": 0,
        "enable_audio": False,
        "language": "English",
        "custom_persona": "Here enter your custom persona",
    }
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# Load config
try:
    config = load_config()
except Exception as e:
    st.error(f"Failed to load configuration: {str(e)}")
    st.stop()

# Groq API setup
try:
    client = get_groq_client(config['api_key'])
except Exception as e:
    st.error(f"Failed to create Groq API clients: {str(e)}")
    st.stop()

# Function to export chat
def export_chat(format):
    chat_history = "\n\n".join([f"**{m['role'].capitalize()}:** {m['content']}" for m in st.session_state.messages])
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
            st.download_button("Download PDF", f, file_name=filename)

# Function to process uploaded file
def process_uploaded_file(uploaded_file):
    file_type_handlers = {
        "application/pdf": lambda f: " ".join(page.extract_text() for page in PyPDF2.PdfReader(f).pages),
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": lambda f: " ".join(paragraph.text for paragraph in docx.Document(f).paragraphs),
        "text/plain": lambda f: f.getvalue().decode("utf-8"),
        "text/markdown": lambda f: f.getvalue().decode("utf-8"),
        "image/": lambda f: pytesseract.image_to_string(Image.open(f))
    }
    for file_type, handler in file_type_handlers.items():
        if uploaded_file.type.startswith(file_type):
            return handler(uploaded_file)
    raise ValueError("Unsupported file type")

# Function to summarize file
def summarize_file(prompt):
    if st.session_state.file_content:
        full_prompt = f"{prompt}\n\nContent:\n{st.session_state.file_content[:4000]}..."
        st.session_state.messages.extend([
            {"role": "user", "content": f"Summarize the uploaded file: {prompt}"},
            {"role": "assistant", "content": "Certainly! I'll summarize the file content based on your prompt."}
        ])

        try:
            summary = ""
            for chunk in stream_llm_response(client, st.session_state.model_params, [{"role": "user", "content": full_prompt}]):
                summary += chunk
            st.session_state.messages.append({"role": "assistant", "content": summary})
            update_token_count(len(summary.split()))
        except Exception as e:
            st.error(f"Error summarizing file: {str(e)}")
    else:
        st.warning("Please upload a file first.")

# Function to save chat history
def save_chat_history():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    st.session_state.chat_histories.append({
        "name": f"Chat {timestamp}",
        "messages": st.session_state.messages.copy()
    })

# Function to load chat history
def load_chat_history(selected_history):
    for history in st.session_state.chat_histories:
        if history["name"] == selected_history:
            st.session_state.messages = history["messages"].copy()
            break

# Function to update token count
def update_token_count(tokens):
    st.session_state.total_tokens += tokens
    st.session_state.total_cost += tokens * 0.0001

# Function for text-to-speech
def text_to_speech(text, lang):
    lang_map = {
        "english": "en",
        "tamil": "ta",
        "hindi": "hi"
    }
    lang_code = lang_map.get(lang.lower(), "en")
    tts = gTTS(text=text, lang=lang_code)
    audio_file = "temp_audio.mp3"
    tts.save(audio_file)
    with open(audio_file, "rb") as f:
        audio_bytes = f.read()
    os.remove(audio_file)
    st.session_state.audio_base64 = base64.b64encode(audio_bytes).decode()

# Function to reset all states
def reset_all():
    st.session_state.update({
        "messages": [],
        "total_tokens": 0,
        "total_cost": 0,
        "file_content": "",
        "audio_base64": ""
    })

# Function to translate text
def translate_text(text, target_lang):
    if target_lang == "English":
        return text
    translator = GoogleTranslator(source='auto', target=target_lang)
    return translator.translate(text)

# Main function
def main():
    st.markdown("""<h1 style="text-align: center; color: #6ca395;"<i>NesiGroq Chat</i> 💬</h1>""", unsafe_allow_html=True)

    with st.sidebar:
        st.title("🔧 Settings")

        col1, col2 = st.columns(2)
        with col1:
            st.button("📄 Exp.to PDF", on_click=export_chat, args=("pdf",))
            if st.button("Reset All"):
                reset_all()
        with col2:
            st.button("📄 Exp.to md", on_click=export_chat, args=("md",))
            if st.button("Save Chat"):
                save_chat_history()

        selected_history = st.selectbox("Load Chat", options=[h["name"] for h in st.session_state.chat_histories])
        if st.button("Load"):
            load_chat_history(selected_history)

        st.session_state.enable_audio = st.checkbox("Enable Audio Response", value=False)

        st.session_state.language = st.selectbox("Select Language:", ["English", "Tamil", "Hindi"])

        persona_choice = st.selectbox("Persona:", options=["Default", "Custom"], index=0)

        if persona_choice == "Custom":
            st.session_state.custom_persona = st.text_area("Enter Custom Persona:", height=100)
            st.session_state.persona = st.session_state.custom_persona
        else:
            st.session_state.persona = "Default Persona"
            st.text_area("Persona Content:", value=st.session_state.persona, height=100, disabled=True)

        model_choice = st.selectbox("Choose Model:", options=list(config['models'].keys()), format_func=lambda x: config['models'][x]["name"])

        with st.expander("Advanced Model Parameters"):
            max_tokens = st.slider("Max Tokens:", min_value=512, max_value=config['models'][model_choice]["tokens"], value=4096, step=512)
            model_temp = st.slider("Temperature:", min_value=0.0, max_value=2.0, value=0.7, step=0.1)
            top_p = st.slider("Top-p:", min_value=0.0, max_value=1.0, value=0.9, step=0.1)

        st.session_state.model_params = {
            "model": model_choice,
            "temperature": model_temp,
            "max_tokens": max_tokens,
            "top_p": top_p
        }

        uploaded_file = st.file_uploader("Upload File", type=["pdf", "docx", "txt", "md", "png", "jpg", "jpeg"])
        if uploaded_file:
            try:
                st.session_state.file_content = process_uploaded_file(uploaded_file)
                st.success("File uploaded successfully!")
                st.info("You can now ask questions about the uploaded file in the main chat.")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

        st.write(f"Total Tokens: {st.session_state.total_tokens}")
        st.write(f"Total Cost: ${st.session_state.total_cost:.4f}")

    # Main chat interface
    chat_container = st.container()

    with chat_container:
        # Display chat history in a scrollable area
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Prompt input at the bottom
    prompt = st.chat_input("Fast Chat")

    if prompt:
        try:
            validate_prompt(prompt)
            if st.session_state.file_content:
                prompt = f"Based on the uploaded file content, {prompt}\n\nFile content: {st.session_state.file_content[:4000]}..."
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("assistant"):
                response_container = st.empty()
                full_response = ""
                try:
                    for chunk in stream_llm_response(client, st.session_state.model_params, st.session_state.messages):
                        full_response += chunk
                        response_container.markdown(full_response + "▌")

                    translated_response = translate_text(full_response, st.session_state.language)
                    response_container.markdown(translated_response)
                    st.session_state.messages.append({"role": "assistant", "content": translated_response})
                    update_token_count(len(full_response.split()))

                    if st.session_state.enable_audio:
                        text_to_speech(translated_response, st.session_state.language)
                        st.audio(f"data:audio/mp3;base64,{st.session_state.audio_base64}", format="audio/mp3")

                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")
        except ValueError as e:
            st.error(str(e))

    # Scroll to the bottom of the chat
    st.markdown('<script>window.scrollTo(0,document.body.scrollHeight);</script>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

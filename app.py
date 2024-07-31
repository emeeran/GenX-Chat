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

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

# Set page config at the very beginning
st.set_page_config(page_title="Groq-Chat", page_icon="", layout="wide", initial_sidebar_state="expanded")

# Initialize session state
def initialize_session_state():
    default_values = {
        "messages": [],
        "audio_base64": "",
        "file_content": "",
        "chat_histories": [],
        "persona": "default",
        "user": None,
        "model_params": {
            "model": "llama-3.1-70b-versatile",
            "max_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.9
        },
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

# Groq client setup
def get_groq_client(api_key):
    return groq.Groq(api_key=api_key)

def stream_llm_response(client, params, messages):
    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=params["model"],
            max_tokens=params["max_tokens"],
            temperature=params["temperature"],
            top_p=params["top_p"],
            stream=True
        )
        for chunk in chat_completion:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
    except Exception as e:
        st.error(f"Error in API call: {str(e)}")
        yield "I'm sorry, but I encountered an error while processing your request."

def validate_prompt(prompt):
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty")

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
    
    with open(filename, "rb") as f:
        st.download_button(f"Download {format.upper()}", f, file_name=filename)

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

def save_chat_history():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    st.session_state.chat_histories.append({
        "name": f"Chat {timestamp}",
        "messages": st.session_state.messages.copy()
    })

def load_chat_history(selected_history):
    for history in st.session_state.chat_histories:
        if history["name"] == selected_history:
            st.session_state.messages = history["messages"].copy()
            break

def update_token_count(tokens):
    st.session_state.total_tokens += tokens
    st.session_state.total_cost += tokens * 0.0001  # Adjust cost calculation as needed

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

def reset_all():
    st.session_state.update({
        "messages": [],
        "total_tokens": 0,
        "total_cost": 0,
        "file_content": "",
        "audio_base64": ""
    })

def translate_text(text, target_lang):
    if target_lang == "English":
        return text
    translator = GoogleTranslator(source='auto', target=target_lang)
    return translator.translate(text)

def main():
    st.markdown("""<h1 style="text-align: center; color: #6ca395;"><i>Groq-Chat</i> 💬</h1>""", unsafe_allow_html=True)

    with st.sidebar:
        st.title("🔧 Settings")

        with st.expander("Chat Settings"):
            st.button("Reset All", on_click=reset_all)
            st.button("Save Chat History", on_click=save_chat_history)
            chat_history_names = [history["name"] for history in st.session_state.chat_histories]
            selected_history = st.selectbox("Load Chat History", options=[""] + chat_history_names)
            if selected_history:
                st.button("Load Chat History", on_click=load_chat_history, args=(selected_history,))

        with st.expander("Audio Settings"):
            st.session_state.enable_audio = st.checkbox("Enable Audio Response", value=False)
            st.session_state.language = st.selectbox("Select Language:", ["English", "Tamil", "Hindi"])

        with st.expander("Persona Settings"):
            persona_options = ["default", "Custom"]
            persona_choice = st.selectbox("Persona:", options=persona_options)
            if persona_choice == "Custom":
                st.session_state.custom_persona = st.text_area("Enter Custom Persona:", height=100)
                st.session_state.persona = st.session_state.custom_persona
            else:
                st.session_state.persona = "Default assistant persona"
            st.text_area("Persona Content:", value=st.session_state.persona, height=100, disabled=True)

        with st.expander("Model Settings"):
            st.session_state.model_params["model"] = st.selectbox("Choose Model:", options=["llama-3.1-405b-reasoning", "llama-3.1-70b-versatile", "llama-3.1-8b-instant"])
            st.session_state.model_params["max_tokens"] = st.slider("Max Tokens:", min_value=512, max_value=4096, value=1024, step=512)
            st.session_state.model_params["temperature"] = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
            st.session_state.model_params["top_p"] = st.slider("Top-p:", min_value=0.0, max_value=1.0, value=0.9, step=0.1)

        with st.expander("File Upload"):
            uploaded_file = st.file_uploader("Upload a file", type=["pdf", "docx", "txt", "md", "jpg", "jpeg", "png"])
            if uploaded_file is not None:
                try:
                    st.session_state.file_content = process_uploaded_file(uploaded_file)
                    st.success("File processed successfully")
                except Exception as e:
                    st.error(f"Error processing file: {e}")

        with st.expander("Export Chat"):
            export_format = st.selectbox("Export Format", ["md", "pdf"])
            if st.button("Export Chat"):
                export_chat(export_format)

    # Main chat interface
    chat_container = st.container()

    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Prompt input at the bottom
    prompt = st.chat_input("Enter your message...")

    if prompt:
        try:
            validate_prompt(prompt)
            if st.session_state.file_content:
                prompt = f"Based on the uploaded file content, {prompt}\n\nFile content: {st.session_state.file_content[:4000]}..."
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                for chunk in stream_llm_response(get_groq_client(API_KEY), st.session_state.model_params, st.session_state.messages):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})

            if st.session_state.enable_audio:
                text_to_speech(full_response, st.session_state.language)

            update_token_count(len(full_response.split()))
        except ValueError as e:
            st.error(str(e))

    # Display audio if available
    if st.session_state.audio_base64:
        audio_bytes = base64.b64decode(st.session_state.audio_base64)
        st.audio(audio_bytes, format="audio/mp3")

    # Display statistics
    st.sidebar.markdown("### Statistics")
    st.sidebar.write(f"Total Tokens Used: {st.session_state.total_tokens}")
    st.sidebar.write(f"Total Cost (USD): ${st.session_state.total_cost:.4f}")

if __name__ == "__main__":
    main()
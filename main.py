import streamlit as st
import os
import sys
import base64
from gtts import gTTS
import PyPDF2
import docx
from PIL import Image
import pytesseract
from deep_translator import GoogleTranslator
from textblob import TextBlob
from datetime import datetime
from fpdf import FPDF

from config import load_config
from api_handler import get_groq_client, get_async_groq_client, stream_llm_response, APIError
from utils import play_audio, validate_prompt
from auth import authenticate
from sys_message import system_messages

# Set the page configuration
st.set_page_config(page_title="Groq-Chat", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

# Ensure the current directory is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Load configuration
try:
    config = load_config()
except Exception as e:
    st.error(f"Failed to load configuration: {str(e)}")
    st.stop()

# Initialize session state
for key, default_value in {
    "messages": [],
    "audio_base64": "",
    "file_content": "",
    "chat_histories": [],
    "system_message": system_messages["default"],
    "user": None,
    "model_params": {},
    "total_tokens": 0,
    "total_cost": 0,
    "enable_audio": False,
    "language": "english"
}.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# Groq API setup
try:
    client = get_groq_client(config['api_key'])
    async_client = get_async_groq_client(config['api_key'])
except Exception as e:
    st.error(f"Failed to create Groq API clients: {str(e)}")
    st.stop()

def export_chat(format):
    chat_history = "\n\n".join([f"**{m['role'].capitalize()}:** {m['content']}" for m in st.session_state.messages])
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"chat_exports/chat_history_{timestamp}.{format}"
    os.makedirs("chat_exports", exist_ok=True)

    if format == "md":
        with open(filename, "w") as f:
            f.write(chat_history)
    elif format == "pdf":
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, chat_history)
        pdf.output(filename)
        with open(filename, "rb") as f:
            st.download_button("Download PDF", f, file_name=filename)

def process_uploaded_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        return " ".join(page.extract_text() for page in pdf_reader.pages)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(uploaded_file)
        return " ".join(paragraph.text for paragraph in doc.paragraphs)
    elif uploaded_file.type in ["text/plain", "text/markdown"]:
        return uploaded_file.getvalue().decode("utf-8")
    elif uploaded_file.type.startswith("image/"):
        image = Image.open(uploaded_file)
        return pytesseract.image_to_string(image)
    else:
        raise ValueError("Unsupported file type")

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
        except APIError as e:
            st.error(f"Error summarizing file: {str(e)}")
    else:
        st.warning("Please upload a file first.")

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
    st.session_state.total_cost += tokens * 0.0001  # Adjust as needed

def text_to_speech(text, lang):
    lang_map = {"english": "en", "tamil": "ta", "hindi": "hi"}
    lang_code = lang_map.get(lang.lower(), "en")
    tts = gTTS(text=text, lang=lang_code)
    audio_file = "temp_audio.mp3"
    tts.save(audio_file)
    with open(audio_file, "rb") as f:
        audio_bytes = f.read()
    os.remove(audio_file)
    audio_base64 = base64.b64encode(audio_bytes).decode()
    st.session_state.audio_base64 = audio_base64

def reset_all():
    for key in ["messages", "total_tokens", "total_cost", "file_content", "audio_base64"]:
        st.session_state[key] = ""

def translate_text(text, target_lang):
    if target_lang == "english":
        return text
    translator = GoogleTranslator(source='auto', target=target_lang)
    return translator.translate(text)

def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

def generate_contextual_response(client, model_params, user_input, conversation_history):
    sentiment_score = analyze_sentiment(user_input)
    sentiment = "positive" if sentiment_score > 0 else "negative" if sentiment_score < 0 else "neutral"

    context = " ".join([msg["content"] for msg in conversation_history if msg["role"] == "user"])
    full_prompt = f"Sentiment: {sentiment}\nContext: {context}\nUser: {user_input}\nAssistant:"

    response = ""
    for chunk in stream_llm_response(client, model_params, [{"role": "user", "content": full_prompt}]):
        response += chunk

    return response

def main():
    st.markdown("""<h1 style="text-align: center; color: #6ca395;"> <i>Groq Fast Chat</i> 💬</h1>""", unsafe_allow_html=True)

    with st.sidebar:
        st.title("🔧 Settings")

        col1, col2 = st.columns(2)
        with col1:
            st.button("📄 Save-PDF", on_click=export_chat, args=("pdf",))
            if st.button("Reset All"):
                reset_all()
        with col2:
            st.button("📄 Save-md", on_click=export_chat, args=("md",))
            if st.button("Save Chat"):
                save_chat_history()

        selected_history = st.selectbox("Load Chat", options=[h["name"] for h in st.session_state.chat_histories])
        if st.button("Load"):
            load_chat_history(selected_history)

        st.session_state.enable_audio = st.checkbox("Enable Audio Response", value=False)
        st.session_state.language = st.selectbox("Select Language:", ["English", "Tamil", "Hindi"])
        system_message_choice = st.selectbox("System Message:", options=list(system_messages.keys()), index=list(system_messages.keys()).index("default"))
        st.session_state.system_message = system_messages[system_message_choice]

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
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    prompt = st.chat_input("Message Groq Fast Chat...")

    if prompt:
        try:
            validate_prompt(prompt)
            user_input = prompt
            st.session_state.messages.append({"role": "user", "content": user_input})

            with st.chat_message("assistant"):
                response_container = st.empty()
                full_response = ""
                try:
                    context_response = generate_contextual_response(client, st.session_state.model_params, user_input, st.session_state.messages)
                    translated_response = translate_text(context_response, st.session_state.language)
                    response_container.markdown(translated_response)
                    st.session_state.messages.append({"role": "assistant", "content": translated_response})
                    update_token_count(len(context_response.split()))

                    if st.session_state.enable_audio:
                        text_to_speech(translated_response, st.session_state.language)
                        st.audio(f"data:audio/mp3;base64,{st.session_state.audio_base64}", format="audio/mp3")

                except APIError as e:
                    st.error(f"API Error: {str(e)}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")
        except ValueError as e:
            st.error(str(e))

    st.markdown('<script>window.scrollTo(0,document.body.scrollHeight);</script>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()


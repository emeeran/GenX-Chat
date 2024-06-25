import streamlit as st
import os
import sys

# Ensure the current directory is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Set page config at the very beginning
st.set_page_config(page_title="Groq-Chat", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

from datetime import datetime
from fpdf import FPDF
import PyPDF2

# Import from our custom modules
from config import load_config
from api_handler import get_groq_client, get_async_groq_client, stream_llm_response, APIError
from utils import play_audio, validate_prompt
from auth import authenticate
from sys_message import system_messages  # Import system messages

# Load configuration
try:
    config = load_config()
except Exception as e:
    st.error(f"Failed to load configuration: {str(e)}")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "audio_base64" not in st.session_state:
    st.session_state.audio_base64 = ""
if "pdf_content" not in st.session_state:
    st.session_state.pdf_content = ""
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = []
if "system_message" not in st.session_state:
    st.session_state.system_message = system_messages["default"]
if "user" not in st.session_state:
    st.session_state.user = None

# Groq API setup
try:
    client = get_groq_client(config['api_key'])
    async_client = get_async_groq_client(config['api_key'])
except Exception as e:
    st.error(f"Failed to create Groq API clients: {str(e)}")

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

def upload_pdf():
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file:
        try:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            st.session_state.pdf_content = " ".join(page.extract_text() for page in pdf_reader.pages)
            st.success("PDF uploaded successfully!")
        except Exception as e:
            st.error(f"Error uploading PDF: {str(e)}")

def summarize_pdf(client, model_params):
    if st.session_state.pdf_content:
        prompt = f"Please summarize the following PDF content:\n\n{st.session_state.pdf_content[:4000]}..."
        st.session_state.messages.extend([
            {"role": "user", "content": "Summarize the uploaded PDF"},
            {"role": "assistant", "content": "Certainly! I'll summarize the PDF content for you."}
        ])

        try:
            summary = ""
            for chunk in stream_llm_response(client, model_params, [{"role": "user", "content": prompt}]):
                summary += chunk
            st.session_state.messages.append({"role": "assistant", "content": summary})
        except APIError as e:
            st.error(f"Error summarizing PDF: {str(e)}")
    else:
        st.warning("Please upload a PDF first.")

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

def main():
    st.markdown("""<h1 style="text-align: center; color: #6ca395;">🤖 <i>Groq Fast Chat</i> 💬</h1>""", unsafe_allow_html=True)

    with st.sidebar:
        st.title("🔧 Settings")

        if 'total_tokens' in st.session_state:
            st.write(f"Tokens used in last response: {st.session_state.total_tokens}")

        system_message_choice = st.selectbox("System Message:", options=list(system_messages.keys()))
        st.session_state.system_message = system_messages[system_message_choice]

        st.session_state.model_choice = st.selectbox("Choose Model:", options=list(config['models'].keys()), format_func=lambda x: config['models'][x]["name"])
        max_tokens = st.slider("Max Tokens:", min_value=512, max_value=config['models'][st.session_state.model_choice]["tokens"], value=4096, step=512)
        model_temp = st.slider("Temp:", min_value=0.0, max_value=2.0, value=0.7, step=0.1)
        top_p = st.slider("Top-p", min_value=0.0, max_value=1.0, value=0.9, step=0.1)

        model_params = {
            "model": st.session_state.model_choice,
            "temperature": model_temp,
            "max_tokens": max_tokens,
            "top_p": top_p
        }

        col1, col2 = st.columns(2)
        with col1:
            st.button("📄 Exp.to PDF", on_click=export_chat, args=("pdf",))
        with col2:
            st.button("📄 Exp.to md", on_click=export_chat, args=("md",))

        col3, col4 = st.columns(2)
        with col3:
            if st.button("Reset"):
                st.session_state.messages = []
        with col4:
            if st.button("Save Chat"):
                save_chat_history()

        col5, col6 = st.columns(2)
        with col5:
            if st.button("Sum-PDF"):
                summarize_pdf(client, model_params)
        with col6:
            selected_history = st.selectbox("Chat", options=[h["name"] for h in st.session_state.chat_histories])
            if st.button("Load Chat"):
                load_chat_history(selected_history)


        upload_pdf()

    chat_container = st.container()
    prompt = st.chat_input("Hi! How can I help you today?")

    if prompt:
        try:
            validate_prompt(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
        except ValueError as e:
            st.error(str(e))

    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if prompt:
        with st.chat_message("assistant"):
            response_container = st.empty()
            full_response = ""
            try:
                for chunk in stream_llm_response(client, model_params, st.session_state.messages):
                    full_response += chunk
                    response_container.markdown(full_response + "▌")
                response_container.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except APIError as e:
                st.error(f"API Error: {str(e)}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()

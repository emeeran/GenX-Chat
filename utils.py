from gtts import gTTS
from io import BytesIO
import base64

def play_audio(text, voice):
    tts = gTTS(text, lang=voice['lang'], tld=voice['tld'])
    audio_fp = BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    return base64.b64encode(audio_fp.read()).decode()

def validate_prompt(prompt):
    if not prompt or len(prompt.strip()) == 0:
        raise ValueError("Prompt cannot be empty")
    if len(prompt) > 1000:  # Adjust this limit as needed
        raise ValueError("Prompt is too long. Please limit it to 1000 characters.")
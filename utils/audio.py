# utils/audio.py

import base64
import os
from gtts import gTTS

def text_to_speech(text: str, lang: str) -> str:
    lang_map = {"English": "en", "Tamil": "ta", "Hindi": "hi"}
    lang_code = lang_map.get(lang, "en")
    tts = gTTS(text=text, lang=lang_code)
    audio_file = "temp_audio.mp3"
    tts.save(audio_file)
    with open(audio_file, "rb") as f:
        audio_bytes = f.read()
    os.remove(audio_file)
    return base64.b64encode(audio_bytes).decode()
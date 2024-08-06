# utils/text_processing.py

from deep_translator import GoogleTranslator

def word_count(text: str) -> int:
    return len(text.split())

def translate_text(text: str, target_lang: str) -> str:
    if target_lang == "English":
        return text
    translator = GoogleTranslator(source="auto", target=target_lang)
    return translator.translate(text)
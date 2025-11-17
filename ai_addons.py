# ai_addons.py — FINAL VERSION (NO OPENAI)

from deep_translator import GoogleTranslator

def translate_text(text, target_language):
    if not text:
        return ""

    try:
        translated = GoogleTranslator(source="auto", target=target_language.lower()).translate(text)
        return translated
    except Exception as e:
        print("Translation error:", e)
        return text

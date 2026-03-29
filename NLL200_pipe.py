

import re
import gradio as gr
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

LANG_MAP = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "mr": "mar_Deva",
    "sa": "san_Deva",
    "bn": "ben_Beng",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "kn": "kan_Knda",
    "gu": "guj_Gujr",
    "pa": "pan_Guru",
    "ur": "urd_Arab",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
}

LANG_NAMES = {
    "en": "English", "hi": "Hindi", "mr": "Marathi", "sa": "Sanskrit",
    "bn": "Bengali", "ta": "Tamil", "te": "Telugu", "kn": "Kannada",
    "gu": "Gujarati", "pa": "Punjabi", "ur": "Urdu",
    "fr": "French", "de": "German", "es": "Spanish",
}

INTENT_DATA = [
    ("hello", "greeting"), ("hi", "greeting"), ("hey", "greeting"),
    ("good morning", "greeting"), ("good evening", "greeting"),
    ("namaste", "greeting"), ("greetings", "greeting"),

    ("i need help", "help"), ("help me", "help"), ("can you help", "help"),
    ("i need assistance", "help"), ("please help", "help"),
    ("support", "help"), ("i am stuck", "help"),

    ("what services do you provide", "service_inquiry"),
    ("what can you do", "service_inquiry"),
    ("tell me about your services", "service_inquiry"),
    ("what do you offer", "service_inquiry"),
    ("how can you assist me", "service_inquiry"),
    ("features", "service_inquiry"),

    ("i am sick", "healthcare"), ("i have a fever", "healthcare"),
    ("symptoms", "healthcare"), ("doctor", "healthcare"),
    ("medicine", "healthcare"), ("health advice", "healthcare"),
    ("i am not feeling well", "healthcare"),

    ("i want to learn", "education"), ("study help", "education"),
    ("teach me", "education"), ("explain this topic", "education"),
    ("homework", "education"), ("course", "education"),

    ("bye", "goodbye"), ("goodbye", "goodbye"), ("see you", "goodbye"),
    ("thank you", "goodbye"), ("thanks", "goodbye"), ("that's all", "goodbye"),
]

RESPONSES = {
    "greeting":        "Hello! Welcome. How can I assist you today?",
    "help":            "Of course! I'm here to help. Please describe your issue.",
    "service_inquiry": "I can help with healthcare info, education, and general support in many languages!",
    "healthcare":      "Please describe your symptoms. Remember, I'm not a doctor — consult a professional for serious concerns.",
    "education":       "Great! I love helping with learning. What topic would you like to explore?",
    "goodbye":         "Goodbye! Take care. Feel free to return anytime.",
    "unknown":         "I'm not sure I understood that. Could you rephrase?",
}

def train_intent_classifier():
    texts, labels = zip(*INTENT_DATA)
    clf = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf",   LogisticRegression(max_iter=500)),
    ])
    clf.fit(texts, labels)
    return clf

intent_clf = train_intent_classifier()

print("Loading NLLB translation model... (first run may take a few minutes)")
NLLB_MODEL_NAME = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print("Model loaded ✓")

def detect_language(text: str) -> str:
    """Detect language code using langdetect."""
    try:
        code = detect(text)
        return code if code in LANG_MAP else "en"
    except Exception:
        return "en"

def clean_text(text: str) -> str:
    """Basic text cleaning before translation."""
    text = text.lower().strip()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^\w\s\u0900-\u097F\u0980-\u09FF"
                  r"\u0A00-\u0A7F\u0B00-\u0B7F"
                  r"\u0B80-\u0BFF\u0C00-\u0C7F"
                  r"\u0C80-\u0CFF\u0A80-\u0AFF]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def translate(text: str, src_lang: str, tgt_lang: str) -> str:
    """Translate text using NLLB-200."""
    if src_lang == tgt_lang:
        return text
    try:
        tokenizer.src_lang = src_lang
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
        generated = model.generate(
            **inputs,
            forced_bos_token_id=bos_token_id,
            max_length=512,
        )
        return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    except Exception as e:
        print(f"Translation error: {e}")
        return text

def classify_intent(text: str) -> str:
    """Classify intent from English text."""
    return intent_clf.predict([text])[0]

def chat_pipeline(user_input: str):
    """
    Full pipeline:
    Input (any language)
      → Language Detection
      → Text Cleaning
      → Translate to English (pivot)
      → Intent Classification
      → Response Generation
      → Translate Response back
      → Output
    """
    log = []

    lang_code = detect_language(user_input)
    lang_name = LANG_NAMES.get(lang_code, lang_code)
    nllb_src  = LANG_MAP.get(lang_code, "eng_Latn")
    log.append(f"🔍 Detected language: {lang_name} ({lang_code})")

    cleaned = clean_text(user_input)
    log.append(f"🧹 Cleaned text: {cleaned}")

    english_text = translate(cleaned, nllb_src, "eng_Latn")
    log.append(f"🌐 Translated to English: {english_text}")

    intent = classify_intent(english_text)
    log.append(f"🎯 Detected intent: {intent}")

    english_response = RESPONSES.get(intent, RESPONSES["unknown"])
    log.append(f"💬 English response: {english_response}")

    if lang_code != "en":
        final_response = translate(english_response, "eng_Latn", nllb_src)
        log.append(f"🔁 Translated back to {lang_name}: {final_response}")
    else:
        final_response = english_response

    return final_response, "\n".join(log)

def gradio_chat(user_input):
    if not user_input.strip():
        return "", "Please enter a message."
    response, pipeline_log = chat_pipeline(user_input)
    return response, pipeline_log

with gr.Blocks(title="Multilingual Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🌏 Multilingual Chatbot
    **Supports:** Hindi • Marathi • Sanskrit • Bengali • Tamil • Telugu • Kannada • Gujarati • Punjabi • Urdu • English • French + more
    
    Type in **any language** — the pipeline auto-detects, translates, understands, and responds in your language.
    """)

    with gr.Row():
        with gr.Column(scale=2):
            user_input = gr.Textbox(
                label="Your Message",
                placeholder='Try: "नमस्ते मुझे मदद चाहिए" or "मला मदत हवी आहे" or "hello"',
                lines=3,
            )
            submit_btn = gr.Button("Send 🚀", variant="primary")
            response_box = gr.Textbox(label="Chatbot Response", lines=3, interactive=False)

        with gr.Column(scale=1):
            pipeline_log = gr.Textbox(
                label="🔬 Pipeline Steps (for evaluation)",
                lines=12,
                interactive=False,
            )

    gr.Examples(
        examples=[
            ["नमस्ते मुझे मदद चाहिए"],
            ["मला मदत हवी आहे"],
            ["नमस्ते अहम् सहायतां इच्छामि"],
            ["আমার সাহায্য দরকার"],
            ["எனக்கு உதவி வேண்டும்"],
            ["నాకు సహాయం కావాలి"],
            ["ನನಗೆ ಸಹಾಯ ಬೇಕು"],
            ["مجھے مدد چاہیے"],
            ["what services do you provide"],
            ["j'ai besoin d'aide"],
        ],
        inputs=user_input,
    )

    submit_btn.click(gradio_chat, inputs=user_input, outputs=[response_box, pipeline_log])
    user_input.submit(gradio_chat, inputs=user_input, outputs=[response_box, pipeline_log])

if __name__ == "__main__":
    demo.launch(share=True)
import streamlit as st
from langdetect import detect
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_translation_model():
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, translation_model = load_translation_model()

intent_model = joblib.load("intent_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

# ---------------- LANGUAGE MAP ----------------
LANG_CODE_MAP = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "mr": "mar_Deva"
}

# ---------------- FUNCTIONS ----------------
def translate(text, src_lang, tgt_lang):
    tokenizer.src_lang = src_lang
    encoded = tokenizer(text, return_tensors="pt")
    generated_tokens = translation_model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
        max_length=128
    )
    return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

def get_response(intent):
    responses = {
        "greeting": "Hello! How can I help you?",
        "help": "Please describe your issue.",
        "service_inquiry": "We provide multilingual chatbot support.",
        "goodbye": "Goodbye!"
    }
    return responses.get(intent, "Sorry, I did not understand that.")

# ---------------- UI ----------------
st.title("Multilingual Chatbot Prototype")

user_input = st.text_input("Enter your message")

if user_input:
    try:
        detected_lang = detect(user_input)
    except:
        detected_lang = "en"

    src_lang = LANG_CODE_MAP.get(detected_lang, "eng_Latn")

    # Translate to English if needed
    if detected_lang != "en":
        text_en = translate(user_input, src_lang, "eng_Latn")
    else:
        text_en = user_input

    # Intent classification
    X_vec = vectorizer.transform([text_en])
    intent = intent_model.predict(X_vec)[0]

    # Response in English
    response_en = get_response(intent)

    # Translate back
    if detected_lang != "en":
        final_response = translate(response_en, "eng_Latn", src_lang)
    else:
        final_response = response_en

    # Output
    st.markdown(f"**Detected Language:** `{detected_lang}`")
    st.markdown(f"**Intent:** `{intent}`")
    st.markdown("**Response:**")
    st.success(final_response)

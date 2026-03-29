import torch
from langdetect import detect, DetectorFactory
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------------
# 1. Language detection (NO fastText)
# -----------------------------
DetectorFactory.seed = 0

def detect_lang(text):
    try:
        return detect(text), 1.0
    except:
        return "en", 0.0

# -----------------------------
# 2. Load translation model
# -----------------------------
MODEL_NAME = "facebook/nllb-200-distilled-600M"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model.to(DEVICE)

# langdetect → NLLB mapping
LANG_MAP = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "fr": "fra_Latn",
    "de": "deu_Latn"
}

def translate(text, src_lang, tgt_lang):
    tokenizer.src_lang = src_lang

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True
    ).to(DEVICE)

    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=256
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)


# -----------------------------
# 3. Chatbot logic (English only)
# -----------------------------
def chatbot_logic(text_en):
    text_en = text_en.lower()

    if "hello" in text_en or "hi" in text_en:
        return "Hello. How can I help you?"
    if "your name" in text_en:
        return "I am a multilingual chatbot."
    if "bye" in text_en:
        return "Goodbye. Have a nice day."

    return "I understand your message."

# -----------------------------
# 4. Full pipeline
# -----------------------------
def chat(user_text):
    src_lang, _ = detect_lang(user_text)

    if src_lang not in LANG_MAP:
        src_lang = "en"

    if src_lang != "en":
        text_en = translate(
            user_text,
            LANG_MAP[src_lang],
            LANG_MAP["en"]
        )
    else:
        text_en = user_text

    reply_en = chatbot_logic(text_en)

    if src_lang != "en":
        reply = translate(
            reply_en,
            LANG_MAP["en"],
            LANG_MAP[src_lang]
        )
    else:
        reply = reply_en

    return reply

# -----------------------------
# 5. CLI loop
# -----------------------------
if __name__ == "__main__":
    print("Multilingual Chatbot. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        print("Bot:", chat(user_input))

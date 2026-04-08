import re
from typing import Any, cast

from langdetect import DetectorFactory
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

try:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    _HF_TRANSLATION_AVAILABLE = True
except Exception:
    torch = cast(Any, None)
    AutoModelForSeq2SeqLM = cast(Any, None)
    AutoTokenizer = cast(Any, None)
    _HF_TRANSLATION_AVAILABLE = False

DetectorFactory.seed = 0

PIVOT_LANGUAGE = "en"
NLLB_MODEL_NAME = "facebook/nllb-200-distilled-600M"

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
}

EXAMPLE_PROMPTS = [
    {"lang": "en", "text": "Hello, please summarize this paragraph."},
    {"lang": "hi", "text": "नमस्ते, क्या आप मेरी मदद कर सकते हैं?"},
    {"lang": "mr", "text": "नमस्कार, मला हा मजकूर संक्षेपात हवा आहे."},
    {"lang": "sa", "text": "नमः, कृपया एतत् संक्षेपेण वद।"},
    {"lang": "bn", "text": "হ্যালো, এই লেখাটি সংক্ষেপে বলুন।"},
    {"lang": "ta", "text": "வணக்கம், இந்த உரையை சுருக்கமாக கூறுங்கள்."},
    {"lang": "te", "text": "హలో, ఈ వాక్యాన్ని సులభంగా వివరించండి."},
    {"lang": "kn", "text": "ಹಲೋ, ಈ ಪಠ್ಯವನ್ನು ಸಂಕ್ಷಿಪ್ತವಾಗಿ ತಿಳಿಸಿ."},
    {"lang": "gu", "text": "નમસ્તે, કૃપા કરીને આ લખાણનું સારાંશ આપો."},
    {"lang": "pa", "text": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਕਿਰਪਾ ਕਰਕੇ ਇਸਨੂੰ ਸਮਝਾਓ।"},
    {"lang": "ur", "text": "ہیلو، براہ کرم اس متن کی وضاحت کریں۔"},
]

REPLY_TEMPLATES = {
    "fallback": {
        "en": 'I understood your request. Could you share a little more detail? (Input: "{text}")',
        "hi": 'मैंने आपकी बात समझ ली। कृपया थोड़ी और जानकारी दें। (इनपुट: "{text}")',
        "mr": 'तुमचा मुद्दा समजला. कृपया थोडी अधिक माहिती द्या. (इनपुट: "{text}")',
        "sa": 'भवतः अभिप्रायः ज्ञातः। कृपया अधिकविवरणं ददातु। (इनपुट: "{text}")',
        "bn": 'আমি আপনার অনুরোধ বুঝেছি। অনুগ্রহ করে আরও কিছু তথ্য দিন। (ইনপুট: "{text}")',
        "ta": 'உங்கள் கோரிக்கையை புரிந்துகொண்டேன். மேலும் சில விவரங்களை பகிரவும். (உள்ளீடு: "{text}")',
        "te": 'మీ అభ్యర్థన నాకు అర్థమైంది. దయచేసి ఇంకొంచెం వివరాలు చెప్పండి. (ఇన్‌పుట్: "{text}")',
        "kn": 'ನಿಮ್ಮ ವಿನಂತಿ ಅರ್ಥವಾಗಿದೆ. ದಯವಿಟ್ಟು ಇನ್ನಷ್ಟು ವಿವರ ನೀಡಿ. (ಇನ್‌ಪುಟ್: "{text}")',
        "gu": 'તમારી વિનંતી સમજાઈ ગઈ. કૃપા કરીને થોડું વધુ વિગત આપો. (ઇનપુટ: "{text}")',
        "pa": 'ਤੁਹਾਡੀ ਬੇਨਤੀ ਸਮਝ ਆ ਗਈ ਹੈ। ਕਿਰਪਾ ਕਰਕੇ ਹੋਰ ਵੇਰਵਾ ਦਿਓ। (ਇਨਪੁੱਟ: "{text}")',
        "ur": 'میں نے آپ کی درخواست سمجھ لی ہے۔ براہ کرم مزید تفصیل دیں۔ (ان پٹ: "{text}")',
    },
    "greeting": {
        "en": 'Hello! I am ready to help. Tell me what you want to do next.',
        "hi": 'नमस्ते! मैं आपकी मदद के लिए तैयार हूं। बताइए आगे क्या करना है।',
        "mr": 'नमस्कार! मी मदतीसाठी तयार आहे. पुढे काय करायचे ते सांगा.',
        "sa": 'नमः! अहं सहाय्यार्थं सज्जः अस्मि। अनन्तरं किं कर्तव्यम् इति वदत।',
        "bn": 'নমস্কার! আমি সাহায্য করতে প্রস্তুত। এখন কী করতে চান বলুন।',
        "ta": 'வணக்கம்! உதவ தயாராக இருக்கிறேன். அடுத்து என்ன செய்ய வேண்டும் என்று சொல்லுங்கள்.',
        "te": 'నమస్తే! మీకు సహాయం చేయడానికి నేను సిద్ధంగా ఉన్నాను. తర్వాత ఏమి చేయాలో చెప్పండి.',
        "kn": 'ನಮಸ್ಕಾರ! ನಿಮಗೆ ಸಹಾಯ ಮಾಡಲು ಸಿದ್ಧನಿದ್ದೇನೆ. ಮುಂದೆ ಏನು ಮಾಡಬೇಕೆಂದು ಹೇಳಿ.',
        "gu": 'નમસ્તે! હું મદદ માટે તૈયાર છું. હવે આગળ શું કરવું તે કહો.',
        "pa": 'ਸਤ ਸ੍ਰੀ ਅਕਾਲ! ਮੈਂ ਮਦਦ ਲਈ ਤਿਆਰ ਹਾਂ। ਹੁਣ ਅੱਗੇ ਕੀ ਕਰਨਾ ਹੈ ਦੱਸੋ।',
        "ur": 'سلام! میں مدد کے لیے تیار ہوں۔ بتائیں آگے کیا کرنا ہے۔',
    },
    "summarize": {
        "en": 'Sure. Send the full text and I will return a short summary with key points.',
        "hi": 'ज़रूर। पूरा पाठ भेजिए, मैं मुख्य बिंदुओं के साथ छोटा सारांश दूंगा।',
        "mr": 'नक्की. पूर्ण मजकूर पाठवा, मी मुख्य मुद्द्यांसह छोटा सारांश देईन.',
        "sa": 'आम्। पूर्णं पाठं प्रेषयतु, अहं मुख्यबिन्दुभिः सह संक्षेपं दास्यामि।',
        "bn": 'অবশ্যই। পুরো লেখা পাঠান, আমি মূল পয়েন্টসহ সংক্ষিপ্ত সারাংশ দেব।',
        "ta": 'நிச்சயம். முழு உரையை அனுப்புங்கள்; முக்கிய அம்சங்களுடன் சுருக்கம் தருகிறேன்.',
        "te": 'తప్పకుండా. పూర్తి పాఠ్యాన్ని పంపండి; ముఖ్యాంశాలతో చిన్న సారాంశం ఇస్తాను.',
        "kn": 'ಖಂಡಿತ. ಪೂರ್ಣ ಪಠ್ಯವನ್ನು ಕಳುಹಿಸಿ; ಮುಖ್ಯ ಅಂಶಗಳೊಂದಿಗೆ ಸಂಕ್ಷಿಪ್ತ ಸಾರಾಂಶ ನೀಡುತ್ತೇನೆ.',
        "gu": 'ચોક્કસ. પૂરું લખાણ મોકલો; મુખ્ય મુદ્દાઓ સાથે ટૂંકું સારાંશ આપીશ.',
        "pa": 'ਜ਼ਰੂਰ। ਪੂਰਾ ਲਿਖਤ ਭੇਜੋ; ਮੈਂ ਮੁੱਖ ਬਿੰਦੂਆਂ ਨਾਲ ਛੋਟਾ ਸਾਰ ਦਿਆਂਗਾ।',
        "ur": 'ضرور۔ مکمل متن بھیجیں، میں اہم نکات کے ساتھ مختصر خلاصہ دوں گا۔',
    },
    "translate": {
        "en": 'I can translate that. Share source text and target language.',
        "hi": 'मैं अनुवाद कर सकता हूं। स्रोत पाठ और लक्ष्य भाषा भेजिए।',
        "mr": 'मी भाषांतर करू शकतो. मूळ मजकूर आणि लक्ष्य भाषा पाठवा.',
        "sa": 'अहं अनुवादं कर्तुं शक्नोमि। मूलपाठं लक्ष्यभाषां च प्रेषयतु।',
        "bn": 'আমি অনুবাদ করতে পারি। উৎস লেখা ও লক্ষ্য ভাষা পাঠান।',
        "ta": 'நான் மொழிபெயர்க்க முடியும். மூல உரையும் இலக்கு மொழியும் அனுப்புங்கள்.',
        "te": 'నేను అనువదించగలను. మూల పాఠ్యం మరియు లక్ష్య భాషను పంపండి.',
        "kn": 'ನಾನು ಅನುವಾದ ಮಾಡಬಹುದು. ಮೂಲ ಪಠ್ಯ ಮತ್ತು ಗುರಿ ಭಾಷೆಯನ್ನು ಕಳುಹಿಸಿ.',
        "gu": 'હું અનુવાદ કરી શકું છું. મૂળ લખાણ અને લક્ષ્ય ભાષા મોકલો.',
        "pa": 'ਮੈਂ ਅਨੁਵਾਦ ਕਰ ਸਕਦਾ ਹਾਂ। ਮੂਲ ਲਿਖਤ ਅਤੇ ਟਾਰਗੇਟ ਭਾਸ਼ਾ ਭੇਜੋ।',
        "ur": 'میں ترجمہ کر سکتا ہوں۔ اصل متن اور ہدف زبان بھیجیں۔',
    },
    "explain": {
        "en": 'I can explain this step by step. Paste the exact part you want simplified.',
        "hi": 'मैं इसे चरण-दर-चरण समझा सकता हूं। जिस भाग को सरल चाहिए वह भेजिए।',
        "mr": 'मी हे टप्प्याटप्प्याने समजावू शकतो. सोपा हवा असलेला भाग पाठवा.',
        "sa": 'अहं क्रमशः व्याख्यातुं शक्नोमि। यत् भागं सरलतया इच्छथ तं प्रेषयतु।',
        "bn": 'আমি ধাপে ধাপে ব্যাখ্যা করতে পারি। যে অংশটি সহজ চান তা পাঠান।',
        "ta": 'இதைக் கட்டம் கட்டமாக விளக்க முடியும். எளிமையாக வேண்டிய பகுதியை அனுப்புங்கள்.',
        "te": 'దశలవారీగా వివరించగలను. సులభంగా కావలసిన భాగాన్ని పంపండి.',
        "kn": 'ಇದನ್ನು ಹಂತ ಹಂತವಾಗಿ ವಿವರಿಸಬಹುದು. ಸರಳವಾಗಿ ಬೇಕಾದ ಭಾಗವನ್ನು ಕಳುಹಿಸಿ.',
        "gu": 'હું આને પગલું-દર-પગલું સમજાવી શકું છું. સરળ સમજાવટ માટેનો ભાગ મોકલો.',
        "pa": 'ਮੈਂ ਇਹਨੂੰ ਕਦਮ-ਦਰ-ਕਦਮ ਸਮਝਾ ਸਕਦਾ ਹਾਂ। ਜਿਹੜਾ ਹਿੱਸਾ ਸੌਖਾ ਚਾਹੀਦਾ ਉਹ ਭੇਜੋ।',
        "ur": 'میں اسے مرحلہ وار سمجھا سکتا ہوں۔ جس حصے کی سادہ وضاحت چاہیے وہ بھیجیں۔',
    },
    "goodbye": {
        "en": 'Goodbye! Come back anytime if you need help.',
        "hi": 'अलविदा! जब भी मदद चाहिए, फिर आइए।',
        "mr": 'नमस्कार! मदत हवी असेल तर कधीही या.',
        "sa": 'नमः। यदा साहाय्यं आवश्यकं भवति तदा पुनरागच्छतु।',
        "bn": 'বিদায়! সাহায্য লাগলে আবার আসুন।',
        "ta": 'பிரியாவிடை! உதவி தேவைப்பட்டால் எப்போது வேண்டுமானாலும் வாருங்கள்.',
        "te": 'వీడ్కోలు! సహాయం కావాలంటే ఎప్పుడైనా తిరిగి రండి.',
        "kn": 'ವಿದಾಯ! ಸಹಾಯ ಬೇಕಾದರೆ ಯಾವಾಗ ಬೇಕಾದರೂ ಮತ್ತೆ ಬನ್ನಿ.',
        "gu": 'આવજો! મદદ જોઈએ તો ક્યારે પણ ફરીથી આવો.',
        "pa": 'ਅਲਵਿਦਾ! ਮਦਦ ਚਾਹੀਦੀ ਹੋਵੇ ਤਾਂ ਕਦੇ ਵੀ ਮੁੜ ਆਓ।',
        "ur": 'خدا حافظ! مدد چاہیے ہو تو کبھی بھی واپس آئیں۔',
    },
}

INTENT_DATASET = [
    {"intent": "greeting", "text": "hello"},
    {"intent": "greeting", "text": "hi there"},
    {"intent": "greeting", "text": "good morning"},
    {"intent": "greeting", "text": "good evening"},
    {"intent": "greeting", "text": "hey chatbot"},
    {"intent": "summarize", "text": "summarize this text"},
    {"intent": "summarize", "text": "give me a summary"},
    {"intent": "summarize", "text": "make this shorter"},
    {"intent": "summarize", "text": "summarize this paragraph"},
    {"intent": "summarize", "text": "short summary please"},
    {"intent": "translate", "text": "translate this to hindi"},
    {"intent": "translate", "text": "can you translate this"},
    {"intent": "translate", "text": "translate to english"},
    {"intent": "translate", "text": "help with translation"},
    {"intent": "translate", "text": "convert this text to another language"},
    {"intent": "explain", "text": "explain this concept"},
    {"intent": "explain", "text": "help me understand this"},
    {"intent": "explain", "text": "explain step by step"},
    {"intent": "explain", "text": "break this down simply"},
    {"intent": "explain", "text": "what does this mean"},
    {"intent": "goodbye", "text": "bye"},
    {"intent": "goodbye", "text": "goodbye"},
    {"intent": "goodbye", "text": "see you later"},
    {"intent": "goodbye", "text": "thanks bye"},
    {"intent": "goodbye", "text": "talk to you later"},
]

_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")
_NON_TEXT_RE = re.compile(r"[^\w\s\u0900-\u097f\u0980-\u09ff\u0a00-\u0aff\u0b80-\u0bff\u0c00-\u0cff\u0600-\u06ff]", flags=re.UNICODE)

_TRANSLATION_RUNTIME: dict[str, Any] | None = None


def preprocess_text(text: str) -> str:
    no_urls = _URL_RE.sub(" ", text)
    cleaned = _NON_TEXT_RE.sub(" ", no_urls)
    normalized = _SPACE_RE.sub(" ", cleaned)
    return normalized.strip()


def _load_translation_runtime() -> dict[str, Any]:
    global _TRANSLATION_RUNTIME

    if _TRANSLATION_RUNTIME is not None:
        return _TRANSLATION_RUNTIME

    if not _HF_TRANSLATION_AVAILABLE:
        _TRANSLATION_RUNTIME = {
            "ready": False,
            "backend": "none",
            "reason": "transformers_or_torch_not_installed",
        }
        return _TRANSLATION_RUNTIME

    try:
        tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_NAME)

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = model.to(device)
        model.eval()

        _TRANSLATION_RUNTIME = {
            "ready": True,
            "backend": "nllb",
            "tokenizer": tokenizer,
            "model": model,
            "device": device,
        }
    except Exception as exc:
        _TRANSLATION_RUNTIME = {
            "ready": False,
            "backend": "none",
            "reason": f"load_failed:{type(exc).__name__}",
        }

    return _TRANSLATION_RUNTIME


def _translate_text(text: str, source_lang: str, target_lang: str) -> tuple[str, bool, str]:
    if source_lang == target_lang:
        return text, False, "none"

    runtime = _load_translation_runtime()
    backend = runtime.get("backend", "none")

    if not runtime.get("ready"):
        return text, False, backend

    tokenizer = runtime["tokenizer"]
    model = runtime["model"]
    device = runtime["device"]

    src_token = LANG_MAP.get(source_lang, LANG_MAP[PIVOT_LANGUAGE])
    tgt_token = LANG_MAP.get(target_lang, LANG_MAP[PIVOT_LANGUAGE])

    try:
        tokenizer.src_lang = src_token
        model_inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

        forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_token)
        generated_tokens = model.generate(
            **model_inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=256,
        )
        translated = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        translated = translated.strip() or text
        return translated, True, backend
    except Exception:
        return text, False, backend


def _train_intent_model() -> Pipeline:
    train_texts = [row["text"] for row in INTENT_DATASET]
    train_labels = [row["intent"] for row in INTENT_DATASET]

    model = Pipeline(
        steps=[
            (
                "vectorizer",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 5),
                    min_df=1,
                ),
            ),
            ("classifier", LogisticRegression(max_iter=1500, random_state=42)),
        ]
    )
    model.fit(train_texts, train_labels)
    return model


INTENT_MODEL = _train_intent_model()


def _detect_by_script(text: str) -> str | None:
    if re.search(r"[\u0980-\u09ff]", text):
        return "bn"
    if re.search(r"[\u0b80-\u0bff]", text):
        return "ta"
    if re.search(r"[\u0c00-\u0c7f]", text):
        return "te"
    if re.search(r"[\u0c80-\u0cff]", text):
        return "kn"
    if re.search(r"[\u0a80-\u0aff]", text):
        return "gu"
    if re.search(r"[\u0a00-\u0a7f]", text):
        return "pa"
    if re.search(r"[\u0600-\u06ff]", text):
        return "ur"
    if re.search(r"[\u0900-\u097f]", text):
        if re.search(r"(मला|आहे|हा|संक्षेपात)", text):
            return "mr"
        if re.search(r"(कृपया|एतत्|संक्षेपेण|वद)", text):
            return "sa"
        return "hi"
    return None


def detect_language_code(text: str) -> str:
    clean_text = text.strip()
    if not clean_text:
        return "en"

    script_guess = _detect_by_script(clean_text)
    if script_guess in LANG_MAP:
        return script_guess

    try:
        detected = detect(clean_text)
    except Exception:
        detected = "en"

    if detected in LANG_MAP:
        return detected

    return "en"


def predict_intent(text: str) -> tuple[str, float]:
    clean_text = preprocess_text(text)
    if not clean_text:
        return "fallback", 0.0

    predicted_intent = INTENT_MODEL.predict([clean_text])[0]
    probabilities = INTENT_MODEL.predict_proba([clean_text])[0]
    class_order = INTENT_MODEL.named_steps["classifier"].classes_
    confidence = 0.0

    for idx, class_name in enumerate(class_order):
        if class_name == predicted_intent:
            confidence = float(probabilities[idx])
            break

    if confidence < 0.35:
        return "fallback", confidence

    return str(predicted_intent), confidence


def _get_response_template(intent: str, lang: str) -> str:
    intent_map = REPLY_TEMPLATES.get(intent, REPLY_TEMPLATES["fallback"])
    return intent_map.get(lang, intent_map["en"])


def build_same_language_reply(text: str, language_code: str | None = None) -> str:
    prediction = predict_chat(text, language_code)
    return prediction["assistant_message"]


def predict_chat(text: str, language_code: str | None = None) -> dict[str, Any]:
    clean_text = text.strip()
    preprocessed_text = preprocess_text(clean_text)

    lang = language_code or detect_language_code(clean_text)

    pivot_input = preprocessed_text
    translated_to_pivot = False
    translation_backend = "none"

    if lang != PIVOT_LANGUAGE and preprocessed_text:
        pivot_input, translated_to_pivot, translation_backend = _translate_text(
            preprocessed_text,
            source_lang=lang,
            target_lang=PIVOT_LANGUAGE,
        )

    intent, confidence = predict_intent(pivot_input)

    english_template = _get_response_template(intent, PIVOT_LANGUAGE)
    pivot_response = english_template.format(text=clean_text)

    assistant_message = pivot_response
    translated_from_pivot = False

    if lang != PIVOT_LANGUAGE:
        translated_output, translated_from_pivot, translation_backend = _translate_text(
            pivot_response,
            source_lang=PIVOT_LANGUAGE,
            target_lang=lang,
        )

        if translated_from_pivot:
            assistant_message = translated_output
        else:
            native_template = _get_response_template(intent, lang)
            assistant_message = native_template.format(text=clean_text)

    return {
        "detected_lang": lang,
        "preprocessed_text": preprocessed_text,
        "pivot_text": pivot_input,
        "pivot_response": pivot_response,
        "translated_to_pivot": translated_to_pivot,
        "translated_from_pivot": translated_from_pivot,
        "translation_backend": translation_backend,
        "intent": intent,
        "confidence": round(confidence, 4),
        "assistant_message": assistant_message,
    }
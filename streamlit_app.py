"""Streamlit interface for the multilingual chatbot pipeline.

Run with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from dataset_utils import load_config
from pipeline import LanguageDetector, MultilingualChatbotPipeline, TextPreprocessor


@st.cache_data(show_spinner=False)
def get_config() -> dict:
    return load_config("config.json")


@st.cache_data(show_spinner=False)
def get_metadata() -> dict:
    config = get_config()
    metadata_path = Path(config["runtime"].get("artifacts_dir", "artifacts")) / "metadata.json"
    if not metadata_path.exists():
        return {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def humanize_intent(intent: str) -> str:
    clean = intent.replace("atis_", "").replace("_", " ").strip()
    return clean or intent


@st.cache_resource(show_spinner=False)
def get_translator():
    try:
        from googletrans import Translator

        return Translator()
    except Exception:
        return None


def translate_text(text: str, target_lang: str | None) -> str:
    if not target_lang or target_lang in {"en", "unknown"}:
        return text

    translator = get_translator()
    if translator is None:
        return text

    try:
        return translator.translate(text, dest=target_lang).text
    except Exception:
        return text


def generate_response_variants(result: dict) -> dict:
    intent = result.get("intent", "unknown")
    lang = result.get("detection", {}).get("lang", "unknown")
    confidence = result.get("detection", {}).get("confidence")
    tokens = result.get("preprocessing", {}).get("token_count", 0)
    readable_intent = humanize_intent(intent)

    intent_driven = (
        f"I detected a {readable_intent} request. "
        f"Please provide the missing trip details so I can help."
    )
    contextual = (
        f"Detected intent: {readable_intent}. Language: {lang}. "
        f"Confidence: {confidence if confidence is not None else 'n/a'}. Tokens: {tokens}."
    )
    fallback = (
        f"I understood your message as a {readable_intent} request. "
        f"Tell me a bit more and I will continue."
    )

    return {
        "intent_driven": translate_text(intent_driven, lang),
        "contextual": translate_text(contextual, lang),
        "fallback": translate_text(fallback, lang),
        "source_language": lang,
    }


def build_assistant_reply(result: dict, response_style: str) -> str:
    variants = generate_response_variants(result)
    source_language = variants.pop("source_language", "unknown")
    reply = variants.get(response_style, variants["fallback"])
    return reply, source_language


@st.cache_resource(show_spinner=False)
def get_cached_pipeline(classifier_name: str | None = None) -> MultilingualChatbotPipeline:
    from pipeline import Translator
    
    config = get_config()
    metadata = get_metadata()
    runtime_cfg = config["runtime"]
    model_cfg = config["model"]

    artifacts_dir = Path(runtime_cfg.get("artifacts_dir", "artifacts"))
    extractor_path = artifacts_dir / "extractor.joblib"
    classifier_map = metadata.get("classifier_artifacts", {})
    selected_classifier = classifier_name or metadata.get("best_classifier") or model_cfg.get("classifier", "svm")
    classifier_path = Path(classifier_map.get(selected_classifier, artifacts_dir / "classifier.joblib"))

    if not extractor_path.exists() or not classifier_path.exists():
        raise FileNotFoundError(
            "Model artifacts not found. Run: python train_intent_model.py"
        )

    extractor = joblib.load(extractor_path)
    classifier = joblib.load(classifier_path)
    
    # Initialize translator for input → English translation
    translator = Translator(model="google")

    return MultilingualChatbotPipeline(
        detector=LanguageDetector(backend=model_cfg.get("detector_backend", "langdetect")),
        preprocessor=TextPreprocessor(
            tokenizer=model_cfg.get("tokenizer", "whitespace"),
            stopwords=model_cfg.get("stopwords", "nltk"),
            normalize=model_cfg.get("normalize", "regex"),
        ),
        extractor=extractor,
        classifier=classifier,
        translator=translator,
    )


def main() -> None:
    st.set_page_config(page_title="Multilingual Pipeline Demo", page_icon="🧠", layout="wide")
    st.title("Multilingual Chatbot Pipeline")
    st.caption("Config-driven pipeline using dataset-trained model artifacts")

    config = get_config()
    metadata = get_metadata()
    runtime_cfg = config["runtime"]
    model_cfg = config["model"]
    comparison_rows = metadata.get("comparison", [])
    classifier_options = list(metadata.get("classifier_artifacts", {}).keys())
    if not classifier_options:
        classifier_options = [metadata.get("best_classifier") or model_cfg.get("classifier", "svm")]

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    if "response_style" not in st.session_state:
        st.session_state.response_style = "intent_driven"

    with st.sidebar:
        st.header("Runtime")
        st.write(f"Dataset source: {config['dataset'].get('source')}")
        st.write(f"Best classifier: {metadata.get('best_classifier', model_cfg.get('classifier'))}")
        selected_classifier = st.selectbox("Compare classifier", options=classifier_options, index=0)
        st.session_state.response_style = st.selectbox(
            "Reply style",
            options=["intent_driven", "contextual", "fallback"],
            format_func=lambda value: {
                "intent_driven": "Intent-driven",
                "contextual": "Contextual",
                "fallback": "Fallback",
            }[value],
            index=["intent_driven", "contextual", "fallback"].index(st.session_state.response_style),
        )
        st.write(f"Artifacts dir: {runtime_cfg.get('artifacts_dir')}")
        st.info("If artifacts are missing, run: python train_intent_model.py")

    try:
        pipeline = get_cached_pipeline(selected_classifier)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    compare_tab, chat_tab = st.tabs(["Model Comparison", "Chatbot"])

    with compare_tab:
        st.subheader("Intent model comparison")
        if comparison_rows:
            st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True)
        else:
            st.info("Run training first to populate comparison metrics.")

        st.subheader("Response generation comparison")
        sample_text = st.text_input(
            "Sample message for response comparison",
            value="I want to book a flight from Boston to Denver",
        )

        if st.button("Compare response styles", type="secondary"):
            if not sample_text.strip():
                st.warning("Please enter some text before comparing response styles.")
            else:
                with st.spinner("Running comparison..."):
                    sample_result = pipeline.run(sample_text.strip())

                sample_variants = generate_response_variants(sample_result)
                st.caption(f"Replies translated to: {sample_variants.pop('source_language', 'unknown')}")
                col3, col4 = st.columns(2)
                with col3:
                    st.markdown("**Intent-driven**")
                    st.write(sample_variants["intent_driven"])
                with col4:
                    st.markdown("**Contextual**")
                    st.write(sample_variants["contextual"])
                st.markdown("**Fallback**")
                st.write(sample_variants["fallback"])

                with st.expander("Full JSON output"):
                    st.json(sample_result)

    with chat_tab:
        st.subheader("Chat with the bot")
        st.caption("Messages are stored in session state so you can continue the conversation.")

        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        user_text = st.chat_input("Type your message and press Enter")
        if user_text:
            st.session_state.chat_messages.append({"role": "user", "content": user_text})
            with st.spinner("Thinking..."):
                chat_result = pipeline.run(user_text.strip())
                assistant_reply, source_language = build_assistant_reply(
                    chat_result,
                    st.session_state.response_style,
                )
                assistant_reply = f"{assistant_reply}"
                st.session_state.chat_messages.append({"role": "assistant", "content": assistant_reply})

            with st.expander("Last turn details"):
                st.write(f"Detected language: {chat_result.get('detection', {}).get('lang', 'unknown')}")
                st.write(f"Translated reply language: {source_language}")
                st.write(f"Predicted intent: {chat_result.get('intent', 'n/a')}")
                st.json(chat_result)

            st.rerun()


if __name__ == "__main__":
    main()

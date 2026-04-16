"""Simple Streamlit app for intent prediction and multilingual replies.

Run with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from dataset_utils import load_config
from pipeline import LanguageDetector, MultilingualChatbotPipeline, TextPreprocessor, Translator


@st.cache_data(show_spinner=False)
def load_app_config() -> dict:
    # Read config once and reuse it across Streamlit reruns.
    return load_config("config.json")


@st.cache_data(show_spinner=False)
def load_model_metadata() -> dict:
    # Metadata is written by training and includes model scores.
    config = load_app_config()
    metadata_file = Path(config["runtime"].get("artifacts_dir", "artifacts")) / "metadata.json"
    if not metadata_file.exists():
        return {}
    with open(metadata_file, "r", encoding="utf-8") as file:
        return json.load(file)


def clean_intent_name(intent: str) -> str:
    return intent.replace("atis_", "").replace("_", " ").strip() or intent


def build_reply(intent: str, target_lang: str, translator: Translator | None) -> str:
    # Keep base reply simple, then translate if needed.
    english_reply = (
        f"I think this is a {clean_intent_name(intent)} request. "
        "Please share a few more trip details."
    )

    if translator is None or not target_lang or target_lang in {"en", "unknown"}:
        return english_reply

    translated = translator.translate(english_reply, "en", target_lang)
    return translated["translation"]


def compute_bleu_score(candidate_text: str, reference_text: str) -> float:
    import importlib

    sacrebleu_metrics = importlib.import_module("sacrebleu.metrics")
    metric = sacrebleu_metrics.BLEU(effective_order=True)
    return round(float(metric.sentence_score(candidate_text, [reference_text]).score), 2)


@st.cache_resource(show_spinner=False)
def load_pipeline(classifier_name: str | None = None) -> MultilingualChatbotPipeline:
    # Load heavy model objects once (vectorizer, classifier, translator).
    config = load_app_config()
    metadata = load_model_metadata()

    model_config = config["model"]
    artifacts_dir = Path(config["runtime"].get("artifacts_dir", "artifacts"))

    extractor_file = artifacts_dir / "extractor.joblib"
    classifier_map = metadata.get("classifier_artifacts", {})
    selected_classifier = classifier_name or metadata.get("best_classifier") or model_config.get("classifier", "svm")
    classifier_file = Path(classifier_map.get(selected_classifier, artifacts_dir / "classifier.joblib"))

    if not extractor_file.exists() or not classifier_file.exists():
        raise FileNotFoundError("Model artifacts not found. Run: python train_intent_model.py")

    extractor = joblib.load(extractor_file)
    classifier = joblib.load(classifier_file)
    translator = Translator(model="nllb-600M")

    return MultilingualChatbotPipeline(
        detector=LanguageDetector(backend=model_config.get("detector_backend", "langdetect")),
        preprocessor=TextPreprocessor(
            tokenizer=model_config.get("tokenizer", "whitespace"),
            stopwords=model_config.get("stopwords", "nltk"),
            normalize=model_config.get("normalize", "regex"),
        ),
        extractor=extractor,
        classifier=classifier,
        translator=translator,
        reply_templates_path=config["runtime"].get("reply_templates_path", "replies.json"),
        catalog_path=config["runtime"].get("catalog_path", "travel_catalog.csv"),
        use_ollama=bool(config["runtime"].get("use_ollama", False)),
        ollama_model=config["runtime"].get("ollama_model", "llama3"),
    )


def main() -> None:
    st.set_page_config(page_title="Simple Multilingual Chatbot", page_icon="💬", layout="wide")
    st.title("Simple Multilingual Chatbot")
    st.caption("Type in any language. The model predicts intent and replies in your language.")

    config = load_app_config()
    metadata = load_model_metadata()

    classifier_options = list(metadata.get("classifier_artifacts", {}).keys())
    if not classifier_options:
        classifier_options = [metadata.get("best_classifier") or config["model"].get("classifier", "svm")]

    with st.sidebar:
        # Sidebar shows quick runtime info and lets you switch classifier.
        st.header("Settings")
        st.write(f"Dataset source: {config['dataset'].get('source')}")
        st.write(f"Best classifier from training: {metadata.get('best_classifier', 'unknown')}")
        selected_classifier = st.selectbox("Classifier", classifier_options, index=0)

        comparison_rows = metadata.get("comparison", [])
        if comparison_rows:
            st.markdown("### Training Metrics")
            st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True)

    pipeline = load_pipeline(selected_classifier)

    # Read available intent labels from the loaded classifier.
    intent_labels: list[str] = []
    label_encoder = getattr(pipeline.classifier, "_label_encoder", None) or getattr(pipeline.classifier, "_label_enc", None)
    if label_encoder is not None and hasattr(label_encoder, "classes_"):
        intent_labels = [str(label) for label in label_encoder.classes_]

    normal_tab, live_tab = st.tabs(["Normal Chatbot", "Live Accuracy Chatbot"])

    with normal_tab:
        st.subheader("Normal Chatbot")
        st.caption("Standard chatbot mode without manual scoring.")

        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "normal_last_topic" not in st.session_state:
            st.session_state.normal_last_topic = None
        if "normal_last_city" not in st.session_state:
            st.session_state.normal_last_city = None

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        user_text = st.chat_input("Type your message and press Enter", key="normal_chat_input")
        if user_text:
            st.session_state.messages.append({"role": "user", "content": user_text})

            with st.spinner("Thinking..."):
                # Run full NLP pipeline and build a user-facing reply.
                result = pipeline.run(
                    user_text.strip(),
                    conversation_history=st.session_state.messages,
                    previous_topic=st.session_state.normal_last_topic,
                    previous_city=st.session_state.normal_last_city,
                )
                detected_language = result.get("detection", {}).get("lang", "unknown")
                predicted_intent = result.get("intent", "unknown")
                reply_text = result.get("reply") or build_reply(predicted_intent, detected_language, pipeline.translator)
                st.session_state.normal_last_topic = result.get("topic")
                st.session_state.normal_last_city = result.get("preferred_city")
                st.session_state.messages.append({"role": "assistant", "content": reply_text})

            with st.expander("Last message details"):
                st.write(f"Detected language: {detected_language}")
                st.write(f"Predicted intent: {predicted_intent}")
                st.json(result)

            st.rerun()

    with live_tab:
        st.subheader("Live Accuracy Chatbot")
        st.caption("Pick true intent for accuracy, and optional reference reply for BLEU.")

        if "live_messages" not in st.session_state:
            st.session_state.live_messages = []
        if "live_last_topic" not in st.session_state:
            st.session_state.live_last_topic = None
        if "live_last_city" not in st.session_state:
            st.session_state.live_last_city = None
        if "live_total" not in st.session_state:
            st.session_state.live_total = 0
        if "live_correct" not in st.session_state:
            st.session_state.live_correct = 0
        if "live_bleu_total" not in st.session_state:
            st.session_state.live_bleu_total = 0
        if "live_bleu_sum" not in st.session_state:
            st.session_state.live_bleu_sum = 0.0

        col1, col2, col3, col4 = st.columns(4)
        accuracy = (st.session_state.live_correct / st.session_state.live_total * 100.0) if st.session_state.live_total else 0.0
        avg_bleu = (st.session_state.live_bleu_sum / st.session_state.live_bleu_total) if st.session_state.live_bleu_total else 0.0
        col1.metric("Samples Scored", st.session_state.live_total)
        col2.metric("Correct", st.session_state.live_correct)
        col3.metric("Live Accuracy", f"{accuracy:.2f}%")
        col4.metric("Avg BLEU", f"{avg_bleu:.2f}")

        if st.button("Reset Live Accuracy", key="reset_live_accuracy"):
            st.session_state.live_total = 0
            st.session_state.live_correct = 0
            st.session_state.live_bleu_total = 0
            st.session_state.live_bleu_sum = 0.0

        if intent_labels:
            true_intent = st.selectbox(
                "True intent label for next message",
                ["Skip scoring"] + intent_labels,
                index=0,
                key="live_true_intent",
            )
        else:
            true_intent = "Skip scoring"
            st.info("Intent labels are not available from this loaded classifier, so live scoring is disabled.")

        reference_reply = st.text_input(
            "Reference assistant reply for BLEU (optional)",
            value="",
            key="live_reference_reply",
        )

        for message in st.session_state.live_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        live_user_text = st.chat_input("Type your message and press Enter", key="live_chat_input")
        if live_user_text:
            st.session_state.live_messages.append({"role": "user", "content": live_user_text})

            with st.spinner("Thinking..."):
                live_result = pipeline.run(
                    live_user_text.strip(),
                    conversation_history=st.session_state.live_messages,
                    previous_topic=st.session_state.live_last_topic,
                    previous_city=st.session_state.live_last_city,
                )
                detected_language = live_result.get("detection", {}).get("lang", "unknown")
                predicted_intent = live_result.get("intent", "unknown")
                reply_text = live_result.get("reply") or build_reply(predicted_intent, detected_language, pipeline.translator)
                st.session_state.live_last_topic = live_result.get("topic")
                st.session_state.live_last_city = live_result.get("preferred_city")

                score_note = ""
                if true_intent != "Skip scoring":
                    st.session_state.live_total += 1
                    is_correct = predicted_intent == true_intent
                    if is_correct:
                        st.session_state.live_correct += 1
                    score_note = f"\n\nScoring: predicted={predicted_intent}, true={true_intent}, correct={is_correct}"

                bleu_note = ""
                if reference_reply.strip():
                    bleu_value = compute_bleu_score(reply_text, reference_reply.strip())
                    st.session_state.live_bleu_total += 1
                    st.session_state.live_bleu_sum += bleu_value
                    bleu_note = f"\nBLEU={bleu_value}"

                st.session_state.live_messages.append({"role": "assistant", "content": reply_text + score_note + bleu_note})

            with st.expander("Last message details"):
                st.write(f"Detected language: {detected_language}")
                st.write(f"Predicted intent: {predicted_intent}")
                if true_intent != "Skip scoring":
                    st.write(f"True intent: {true_intent}")
                if reference_reply.strip():
                    st.write(f"Reference reply provided: {reference_reply}")
                st.json(live_result)

            st.rerun()


if __name__ == "__main__":
    main()

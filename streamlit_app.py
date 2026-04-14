"""Simple Streamlit chatbot UI.

Run with:
    streamlit run streamlit_app.py
"""

from pathlib import Path
import json

import joblib
import streamlit as st

from pipeline import MultilingualChatbotPipeline


@st.cache_data(show_spinner=False)
def load_config() -> dict:
    with open("config.json", "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_resource(show_spinner=False)
def load_pipeline() -> MultilingualChatbotPipeline:
    config = load_config()
    artifacts_dir = Path(config["runtime"]["artifacts_dir"])
    extractor = joblib.load(artifacts_dir / "extractor.joblib")
    classifier = joblib.load(artifacts_dir / "classifier.joblib")
    reply_templates_path = config["runtime"]["reply_templates_path"]
    return MultilingualChatbotPipeline(
        extractor=extractor,
        classifier=classifier,
        reply_templates_path=reply_templates_path,
        catalog_path=config["runtime"]["catalog_path"],
        use_ollama=config["runtime"]["use_ollama"],
        ollama_model=config["runtime"]["ollama_model"],
    )


def main() -> None:
    st.set_page_config(page_title="Simple Chatbot", page_icon="💬", layout="wide")
    st.title("Simple Multilingual Chatbot")
    st.caption("Input language is detected, the text is translated to English, then the reply is translated back.")

    pipeline = load_pipeline()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_topic" not in st.session_state:
        st.session_state.last_topic = None
    if "last_city" not in st.session_state:
        st.session_state.last_city = None

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_text = st.chat_input("Type your message and press Enter")
    if user_text:
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.spinner("Thinking..."):
            result = pipeline.run(
                user_text,
                conversation_history=st.session_state.messages,
                previous_topic=st.session_state.last_topic,
                previous_city=st.session_state.last_city,
            )
            reply = result["reply"]
            st.session_state.last_topic = result.get("topic")
            st.session_state.last_city = result.get("preferred_city")
            st.session_state.messages.append({"role": "assistant", "content": reply})

        with st.expander("Last turn details"):
            st.write(f"Detected language: {result['detection']['lang']}")
            st.write(f"Predicted intent: {result['intent']}")
            if result.get("conversation_history_text"):
                st.code(result["conversation_history_text"])
            st.json(result)

        st.rerun()


if __name__ == "__main__":
    main()

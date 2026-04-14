"""Simple multilingual chatbot pipeline.

Flow:
1. Detect language
2. Translate input to English
3. Clean text
4. TF-IDF features
5. Intent classification
6. Build reply
7. Translate reply back to the input language
"""

from pathlib import Path
import csv
import json
import re
import time
import urllib.request

from langdetect import detect, DetectorFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from googletrans import Translator as GoogleTranslator

DetectorFactory.seed = 42


class LanguageDetector:
    def detect(self, text: str) -> dict:
        start = time.perf_counter()
        lang = detect(text)
        latency_ms = (time.perf_counter() - start) * 1000
        return {"lang": lang, "latency_ms": round(latency_ms, 2)}


class TextPreprocessor:
    def process(self, text: str) -> dict:
        start = time.perf_counter()
        text = text.lower()
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"[^\w\s]", " ", text)
        tokens = [token for token in text.split() if len(token) > 1]
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "original": text,
            "tokens": tokens,
            "token_count": len(tokens),
            "latency_ms": round(latency_ms, 2),
        }


class FeatureExtractor:
    def __init__(self, max_features: int = 20000):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
        # Keep the old attribute name so previously saved artifacts still work.
        self._vectorizer = self.vectorizer

    def fit(self, texts: list[str]):
        self.vectorizer.fit(texts)
        self._vectorizer = self.vectorizer

    def transform(self, texts: list[str]):
        start = time.perf_counter()
        vectorizer = getattr(self, "vectorizer", None)
        if vectorizer is None:
            vectorizer = getattr(self, "_vectorizer")
        vectors = vectorizer.transform(texts)
        latency_ms = (time.perf_counter() - start) * 1000
        return vectors, round(latency_ms, 2)


class IntentClassifier:
    def __init__(self):
        self._model = LogisticRegression(max_iter=1000)
        self._label_enc = LabelEncoder()

    def fit(self, vectors, intents: list[str]):
        encoded_intents = self._label_enc.fit_transform(intents)
        self._model.fit(vectors, encoded_intents)

    def predict(self, vectors) -> str:
        prediction = self._model.predict(vectors)[0]
        return self._label_enc.inverse_transform([prediction])[0]


class Translator:
    def __init__(self):
        self.client = GoogleTranslator()

    def translate(self, text: str, source_language: str, target_language: str) -> str:
        result = self.client.translate(text, src=source_language, dest=target_language)
        return result.text


class OllamaResponder:
    def __init__(self, model: str = "llama3", url: str = "http://localhost:11434/api/generate"):
        self.model = model
        self.url = url

    def reply(self, english_text: str, intent: str, history_text: str, catalog_text: str) -> str:
        prompt = (
            "You are a travel assistant.\n"
            "Write only in simple English.\n"
            "Give a helpful short answer.\n\n"
            f"Conversation history:\n{history_text}\n\n"
            f"Available travel data:\n{catalog_text}\n\n"
            f"User message in English: {english_text}\n"
            f"Predicted intent: {intent}\n"
            "Reply:"
        )
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }).encode("utf-8")
        request = urllib.request.Request(self.url, data=payload, headers={"Content-Type": "application/json"})
        response = urllib.request.urlopen(request)
        data = json.loads(response.read().decode("utf-8"))
        return data["response"].strip()


class ReplyGenerator:
    def __init__(self, templates_path: str = "replies.json"):
        with open(Path(templates_path), "r", encoding="utf-8") as file:
            self.templates = json.load(file)

    def build(self, intent: str) -> str:
        if intent in self.templates:
            return self.templates[intent]
        if intent == "hotel":
            return "I can help with hotel booking. Please share the city, dates, and budget."
        return "I can help with that. Please share a few more details."


class TravelCatalog:
    def __init__(self, catalog_path: str = "travel_catalog.csv"):
        self.rows = []
        with open(Path(catalog_path), "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                self.rows.append(row)

    def find_rows(self, text: str, intent: str) -> list[dict]:
        lower_text = text.lower()
        if "hotel" in lower_text:
            wanted_type = "hotel"
        elif "flight" in lower_text or "book" in lower_text or "plane" in lower_text:
            wanted_type = "flight"
        else:
            wanted_type = intent

        query = (text + " " + intent).lower().split()
        matches = []
        for row in self.rows:
            row_type = (row.get("type", "") or "").lower()
            if wanted_type and row_type != wanted_type:
                continue

            row_text = " ".join([
                row.get("type", "") or "",
                row.get("name", "") or "",
                row.get("from_city", "") or "",
                row.get("to_city", "") or "",
                row.get("city", "") or "",
                row.get("notes", "") or "",
            ]).lower()
            if wanted_type in row_text:
                matches.append(row)
                continue

            for word in query:
                if word and word in row_text:
                    matches.append(row)
                    break
        return matches[:10]

    def to_text(self, rows: list[dict]) -> str:
        lines = []
        for row in rows:
            lines.append(
                f"{row.get('type', '') or ''} | {row.get('name', '') or ''} | from {row.get('from_city', '') or ''} | to {row.get('to_city', '') or ''} | city {row.get('city', '') or ''} | date {row.get('date', '') or ''} | time {row.get('time', '') or ''} | price {row.get('price', '') or ''} | available {row.get('availability', '') or ''}"
            )
        return "\n".join(lines)

    def detect_city(self, text: str) -> str | None:
        lower_text = text.lower()
        for row in self.rows:
            city = (row.get("city", "") or "").strip()
            if city and city.lower() in lower_text:
                return city
        return None


class MultilingualChatbotPipeline:
    def __init__(self, extractor=None, classifier=None, reply_templates_path: str = "replies.json", catalog_path: str = "travel_catalog.csv", use_ollama: bool = False, ollama_model: str = "llama3"):
        self.detector = LanguageDetector()
        self.preprocessor = TextPreprocessor()
        self.extractor = extractor if extractor is not None else FeatureExtractor()
        self.classifier = classifier if classifier is not None else IntentClassifier()
        self.translator = Translator()
        self.reply_generator = ReplyGenerator(reply_templates_path)
        self.catalog = TravelCatalog(catalog_path)
        self.use_ollama = use_ollama
        self.ollama = OllamaResponder(ollama_model)

    @staticmethod
    def clean_llm_reply(reply_english: str, intent: str) -> str:
        lower_reply = reply_english.lower()
        has_placeholder = (
            "[" in reply_english and "]" in reply_english
        ) or ("insert" in lower_reply)

        if not has_placeholder:
            return reply_english

        if intent == "hotel":
            return "Please share your check-in and check-out dates so I can complete the hotel booking."
        if intent == "flight":
            return "Please share your travel date so I can complete the flight booking."
        return "Please share the missing details so I can complete your request."

    @staticmethod
    def detect_topic(text: str) -> str | None:
        lower_text = text.lower()
        hotel_words = ["hotel", "room", "stay", "हॉटेल", "रूम", "राहायचे"]
        flight_words = ["flight", "plane", "air", "उड्डाण", "फ्लाइट", "तिकीट"]

        for word in hotel_words:
            if word in lower_text:
                return "hotel"
        for word in flight_words:
            if word in lower_text:
                return "flight"
        return None

    def resolve_topic(self, text: str, english_text: str, history_text: str) -> str | None:
        current_topic = self.detect_topic(text)
        if current_topic:
            return current_topic

        current_topic_en = self.detect_topic(english_text)
        if current_topic_en:
            return current_topic_en

        history_topic = self.detect_topic(history_text)
        if history_topic:
            return history_topic

        return None

    @staticmethod
    def build_history_text(conversation_history: list | None, max_turns: int = 6) -> str:
        if not conversation_history:
            return ""

        lines = []
        for message in conversation_history[-max_turns:]:
            role = message.get("role", "")
            content = message.get("content", "")
            if content:
                lines.append(f"{role.title()}: {content}")
        return "\n".join(lines)

    def fit(self, texts: list[str], intents: list[str]):
        self.extractor.fit(texts)
        vectors, _ = self.extractor.transform(texts)
        self.classifier.fit(vectors, intents)

    def run(self, text: str, conversation_history: list | None = None, previous_topic: str | None = None, previous_city: str | None = None) -> dict:
        result = {}
        history_text = self.build_history_text(conversation_history)
        result["conversation_history_text"] = history_text

        detection = self.detector.detect(text)
        result["detection"] = detection
        source_language = detection["lang"]

        if source_language == "en":
            english_text = text
        else:
            english_text = self.translator.translate(text, source_language, "en")
            result["input_translation"] = english_text

        preprocessing = self.preprocessor.process(english_text)
        result["preprocessing"] = preprocessing

        topic = self.resolve_topic(text, english_text, history_text)
        if not topic and previous_topic:
            topic = previous_topic
        result["topic"] = topic

        preferred_city = self.catalog.detect_city(english_text)
        if not preferred_city:
            preferred_city = self.catalog.detect_city(history_text)
        if not preferred_city and previous_city:
            preferred_city = previous_city
        result["preferred_city"] = preferred_city

        if topic == "hotel":
            intent = "hotel"
        else:
            vectors, _ = self.extractor.transform([" ".join(preprocessing["tokens"] )])
            intent = self.classifier.predict(vectors)

        result["intent"] = intent

        if topic == "hotel":
            catalog_rows = self.catalog.find_rows("hotel " + text, "hotel")
        elif topic == "flight":
            catalog_rows = self.catalog.find_rows("flight " + text, "flight")
        else:
            catalog_rows = self.catalog.find_rows(text, intent)

        if preferred_city:
            city_filtered = [row for row in catalog_rows if (row.get("city", "") or "").lower() == preferred_city.lower()]
            if city_filtered:
                catalog_rows = city_filtered

        catalog_text = self.catalog.to_text(catalog_rows)
        result["catalog_text"] = catalog_text

        if self.use_ollama:
            reply_english = self.ollama.reply(english_text, intent, history_text, catalog_text)
            reply_english = self.clean_llm_reply(reply_english, intent)
            reply_detect = self.detector.detect(reply_english)
            if reply_detect["lang"] != "en":
                reply_english = self.translator.translate(reply_english, reply_detect["lang"], "en")
            result["reply_english"] = reply_english
            if source_language == "en":
                result["reply"] = reply_english
            else:
                result["reply"] = self.translator.translate(reply_english, "en", source_language)
        else:
            reply_english = self.reply_generator.build(intent)
            result["reply_english"] = reply_english

            if source_language == "en":
                result["reply"] = reply_english
            else:
                result["reply"] = self.translator.translate(reply_english, "en", source_language)

        return result


if __name__ == "__main__":
    print("This file contains the simple chatbot pipeline.")

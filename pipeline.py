"""Simple pipeline for multilingual intent classification.

This file keeps the same public class names used by the app and training script,
but the implementation is intentionally straightforward and easier to read.
"""

from __future__ import annotations

import re
import time
from typing import Any


class LanguageDetector:
    """Detect language code from user text."""

    def __init__(self, backend: str = "langdetect", model_path: str | None = None):
        self.backend = backend
        self.model_path = model_path
        self._detect_fn = None
        self._fasttext_model = None
        self._setup_backend()

    def _setup_backend(self) -> None:
        if self.backend != "langdetect":
            raise ValueError("Only default langdetect backend is supported.")

        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 42
        self._detect_fn = detect

    def detect(self, text: str) -> dict:
        start_time = time.perf_counter()
        language = self._detect_fn(text)
        confidence = None

        latency_ms = (time.perf_counter() - start_time) * 1000
        return {"lang": language, "confidence": confidence, "latency_ms": round(latency_ms, 2)}


class TextPreprocessor:
    """Clean user text and split it into tokens for feature extraction."""

    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
    PUNCT_PATTERN = re.compile(r"[^\w\s]")

    def __init__(
        self,
        tokenizer: str = "whitespace",
        stopwords: str = "nltk",
        normalize: str = "regex",
        lang: str = "english",
        custom_stopwords: list[str] | None = None,
    ):
        self.tokenizer = tokenizer
        self.stopwords_mode = stopwords
        self.normalize_mode = normalize
        self.language = lang
        self.custom_stopwords = custom_stopwords or []
        self.stopwords = self._load_stopwords()

    def _load_stopwords(self) -> set[str]:
        if self.stopwords_mode == "none":
            return set()
        if self.stopwords_mode == "custom":
            return {word.lower() for word in self.custom_stopwords}

        import nltk
        from nltk.corpus import stopwords

        nltk.download("stopwords", quiet=True)
        return set(stopwords.words(self.language))

    def _normalize_text(self, text: str) -> str:
        cleaned = text
        if self.normalize_mode in {"regex", "full"}:
            cleaned = self.URL_PATTERN.sub(" ", cleaned)
            cleaned = self.PUNCT_PATTERN.sub(" ", cleaned)
        if self.normalize_mode == "full":
            cleaned = cleaned.lower()
        return " ".join(cleaned.split())

    def _tokenize(self, text: str) -> list[str]:
        # We keep this simple on purpose.
        if self.tokenizer not in {"whitespace", "spacy", "sentencepiece"}:
            return text.split()
        return text.split()

    def process(self, text: str) -> dict:
        start_time = time.perf_counter()
        normalized = self._normalize_text(text)
        tokens = self._tokenize(normalized)
        tokens = [token for token in tokens if len(token) > 1 and token.lower() not in self.stopwords]
        latency_ms = (time.perf_counter() - start_time) * 1000

        return {
            "original": text,
            "normalized": normalized,
            "tokens": tokens,
            "token_count": len(tokens),
            "latency_ms": round(latency_ms, 2),
        }


class FeatureExtractor:
    """Convert cleaned text into numeric vectors for the classifier."""

    def __init__(self, method: str = "tfidf", max_features: int = 20000, model_name: str = "LaBSE"):
        self.method = method
        self.max_features = max_features
        self.model_name = model_name
        self._vectorizer = None
        if self.method != "tfidf":
            raise ValueError("Only tfidf feature extraction is supported.")

    def fit(self, corpus: list[str]) -> "FeatureExtractor":
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self._vectorizer.fit(corpus)
        return self

    def transform(self, texts: list[str]) -> tuple[Any, float]:
        start_time = time.perf_counter()

        if self._vectorizer is None:
            raise RuntimeError("Call fit() before transform()")
        vectors = self._vectorizer.transform(texts)

        latency_ms = (time.perf_counter() - start_time) * 1000
        return vectors, round(latency_ms, 2)

    def fit_transform(self, corpus: list[str]) -> tuple[Any, float]:
        self.fit(corpus)
        return self.transform(corpus)


class IntentClassifier:
    """Train and run one of the supported sklearn intent models."""

    CLASSIFIERS = {
        "svm": ("sklearn.svm", "LinearSVC", {"C": 1.0, "max_iter": 2000}),
        "logreg": ("sklearn.linear_model", "LogisticRegression", {"max_iter": 1000, "C": 5.0}),
        "naive_bayes": ("sklearn.naive_bayes", "MultinomialNB", {}),
        "random_forest": ("sklearn.ensemble", "RandomForestClassifier", {"n_estimators": 200, "n_jobs": -1}),
    }

    def __init__(self, classifier: str = "svm"):
        if classifier not in self.CLASSIFIERS:
            raise ValueError(f"Unsupported classifier: {classifier}")
        self.classifier_name = classifier
        self._model = None
        self._label_encoder = None
        self._label_enc = None
        self._build_model()

    def _build_model(self) -> None:
        import importlib

        module_name, class_name, kwargs = self.CLASSIFIERS[self.classifier_name]
        module = importlib.import_module(module_name)
        self._model = getattr(module, class_name)(**kwargs)

    def fit(self, vectors: Any, labels: list[str]) -> "IntentClassifier":
        from sklearn.preprocessing import LabelEncoder

        self._label_encoder = LabelEncoder()
        encoded_labels = self._label_encoder.fit_transform(labels)
        self._model.fit(vectors, encoded_labels)
        # Keep old attribute name for joblib backward compatibility.
        self._label_enc = self._label_encoder
        return self

    def predict(self, vectors: Any) -> list[str]:
        label_encoder = getattr(self, "_label_encoder", None) or getattr(self, "_label_enc", None)
        if label_encoder is None:
            raise RuntimeError("Label encoder is missing. Train or reload the classifier.")
        predicted_ids = self._model.predict(vectors)
        return label_encoder.inverse_transform(predicted_ids).tolist()

    def evaluate(self, vectors: Any, true_labels: list[str]) -> dict:
        from sklearn.metrics import accuracy_score, classification_report, f1_score

        predicted_labels = self.predict(vectors)
        return {
            "accuracy": round(float(accuracy_score(true_labels, predicted_labels)), 4),
            "macro_f1": round(float(f1_score(true_labels, predicted_labels, average="macro")), 4),
            "report": classification_report(true_labels, predicted_labels),
        }


class Translator:
    """Translate text using NLLB models."""

    MODEL_IDS = {
        "nllb-600M": "facebook/nllb-200-distilled-600M",
        "nllb-1.3B": "facebook/nllb-200-1.3B",
    }

    NLLB_LANGUAGE_MAP = {
        "en": "eng_Latn",
        "hi": "hin_Deva",
        "mr": "mar_Deva",
        "sa": "san_Deva",
        "fr": "fra_Latn",
        "es": "spa_Latn",
        "de": "deu_Latn",
        "it": "ita_Latn",
        "pt": "por_Latn",
    }

    def __init__(self, model: str = "nllb-600M", device: str = "cpu"):
        self.model_name = model
        self.device = device
        self._model = None
        self._tokenizer = None
        self._load_model()

    def _load_model(self) -> None:
        if self.model_name in self.MODEL_IDS:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            hf_model_name = self.MODEL_IDS[self.model_name]
            self._tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(hf_model_name)
            return

    def _normalize_nllb_code(self, code: str) -> str:
        if not code:
            return "eng_Latn"
        if "_" in code and len(code) >= 8:
            return code
        return self.NLLB_LANGUAGE_MAP.get(code.lower(), "eng_Latn")

    def _target_language_token_id(self, target_language_code: str) -> int:
        # Different tokenizer versions expose this differently.
        if hasattr(self._tokenizer, "lang_code_to_id") and self._tokenizer.lang_code_to_id:
            if target_language_code in self._tokenizer.lang_code_to_id:
                return int(self._tokenizer.lang_code_to_id[target_language_code])

        if hasattr(self._tokenizer, "get_lang_id"):
            return int(self._tokenizer.get_lang_id(target_language_code))

        token_id = self._tokenizer.convert_tokens_to_ids(target_language_code)
        if token_id is None:
            raise ValueError(f"Unsupported target language code: {target_language_code}")
        return int(token_id)

    def _hf_translate(self, text: str, source_code: str, target_code: str) -> str:
        import torch

        self._tokenizer.src_lang = source_code
        encoded = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        bos_token_id = self._target_language_token_id(target_code)

        with torch.no_grad():
            generated = self._model.generate(
                **encoded,
                forced_bos_token_id=bos_token_id,
                max_length=512,
            )
        return self._tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> dict:
        start_time = time.perf_counter()

        if self.model_name in {"nllb-600M", "nllb-1.3B"}:
            source = self._normalize_nllb_code(src_lang)
            target = self._normalize_nllb_code(tgt_lang)
            translated_text = self._hf_translate(text, source, target)
        else:
            raise ValueError(f"Unsupported translator model: {self.model_name}")

        latency_ms = (time.perf_counter() - start_time) * 1000
        return {
            "source": text,
            "translation": translated_text,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "latency_ms": round(latency_ms, 2),
        }


class MultilingualChatbotPipeline:
    """Run language detection, preprocessing, vectorization, classification, and translation."""

    def __init__(
        self,
        detector: LanguageDetector | None = None,
        preprocessor: TextPreprocessor | None = None,
        extractor: FeatureExtractor | None = None,
        classifier: IntentClassifier | None = None,
        translator: Translator | None = None,
    ):
        self.detector = detector or LanguageDetector(backend="langdetect")
        self.preprocessor = preprocessor or TextPreprocessor()
        self.extractor = extractor or FeatureExtractor(method="tfidf")
        self.classifier = classifier
        self.translator = translator

    def run(self, text: str, tgt_lang: str | None = None) -> dict:
        result: dict[str, Any] = {}

        # 1) Detect input language first.
        detection = self.detector.detect(text)
        result["detection"] = detection
        source_language = detection["lang"]

        # 2) Translate to English before classification when needed.
        text_for_model = text
        if source_language not in {"en", "unknown"} and self.translator is not None:
            translation = self.translator.translate(text, source_language, "en")
            text_for_model = translation["translation"]
            result["input_translation"] = translation

        # 3) Clean text and tokenize.
        preprocessing = self.preprocessor.process(text_for_model)
        result["preprocessing"] = preprocessing
        cleaned_text = " ".join(preprocessing["tokens"])

        # 4) Convert text to vectors.
        if self.extractor._vectorizer:
            vectors, feature_latency_ms = self.extractor.transform([cleaned_text])
            shape = list(vectors.shape) if hasattr(vectors, "shape") else "unknown"
            result["features"] = {"shape": shape, "latency_ms": feature_latency_ms}
        else:
            vectors = None

        # 5) Predict intent.
        if self.classifier is not None and vectors is not None:
            result["intent"] = self.classifier.predict(vectors)[0]

        # 6) Optional: translate original text to requested target language.
        if self.translator is not None and tgt_lang:
            result["translation"] = self.translator.translate(text, source_language, tgt_lang)

        return result


if __name__ == "__main__":
    print("Run the app with: streamlit run streamlit_app.py")

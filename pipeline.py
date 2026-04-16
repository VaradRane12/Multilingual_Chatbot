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
        if self.backend == "langdetect":
            from langdetect import DetectorFactory, detect

            DetectorFactory.seed = 42
            self._detect_fn = detect
            return

        if self.backend == "langid":
            import langid

            self._detect_fn = lambda text: langid.classify(text)[0]
            return

        if self.backend == "fasttext":
            import fasttext

            model_file = self.model_path or "lid.176.ftz"
            self._fasttext_model = fasttext.load_model(model_file)
            return

        raise ValueError(f"Unknown language detector backend: {self.backend}")

    def detect(self, text: str) -> dict:
        start_time = time.perf_counter()
        try:
            if self.backend == "fasttext":
                labels, probs = self._fasttext_model.predict(text.replace("\n", " "), k=1)
                language = labels[0].replace("__label__", "")
                confidence = float(probs[0])
            else:
                language = self._detect_fn(text)
                confidence = None
        except Exception:
            language = "unknown"
            confidence = 0.0

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

        try:
            import nltk
            from nltk.corpus import stopwords

            nltk.download("stopwords", quiet=True)
            return set(stopwords.words(self.language))
        except Exception:
            return set()

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
        self._ft_model = None
        self._st_model = None

        if self.method not in {"tfidf", "fasttext_avg", "sentence_transformer"}:
            raise ValueError(f"Unsupported feature method: {self.method}")

    def fit(self, corpus: list[str]) -> "FeatureExtractor":
        if self.method == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=(1, 2),
                sublinear_tf=True,
            )
            self._vectorizer.fit(corpus)
            return self

        if self.method == "fasttext_avg":
            import fasttext

            self._ft_model = fasttext.load_model("cc.en.300.bin")
            return self

        from sentence_transformers import SentenceTransformer

        self._st_model = SentenceTransformer(self.model_name)
        return self

    def transform(self, texts: list[str]) -> tuple[Any, float]:
        import numpy as np

        start_time = time.perf_counter()

        if self.method == "tfidf":
            if self._vectorizer is None:
                raise RuntimeError("Call fit() before transform()")
            vectors = self._vectorizer.transform(texts)
        elif self.method == "fasttext_avg":
            vectors = np.vstack([
                np.mean([self._ft_model.get_word_vector(word) for word in text.split()] or [np.zeros(300)], axis=0)
                for text in texts
            ])
        else:
            vectors = self._st_model.encode(texts, show_progress_bar=False)

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
    """Translate text using IndicTrans2, NLLB, mBART, Opus-MT, or Google translator."""

    MODEL_IDS = {
        "indictrans2": {
            "en_indic": [
                "ai4bharat/indictrans2-en-indic-dist-200M",
                "ai4bharat/indictrans2-en-indic-1B",
            ],
            "indic_en": [
                "ai4bharat/indictrans2-indic-en-dist-200M",
                "ai4bharat/indictrans2-indic-en-1B",
            ],
            "indic_indic": [
                "ai4bharat/indictrans2-indic-indic-dist-320M",
                "ai4bharat/indictrans2-indic-indic-1B",
            ],
        },
        "nllb-600M": "facebook/nllb-200-distilled-600M",
        "nllb-1.3B": "facebook/nllb-200-1.3B",
        "mbart50": "facebook/mbart-large-50-many-to-many-mmt",
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

    INDIC_LANGUAGE_MAP = {
        "en": "eng_Latn",
        "hi": "hin_Deva",
        "mr": "mar_Deva",
        "sa": "san_Deva",
        "bn": "ben_Beng",
        "gu": "guj_Gujr",
        "kn": "kan_Knda",
        "ml": "mal_Mlym",
        "or": "ory_Orya",
        "pa": "pan_Guru",
        "ta": "tam_Taml",
        "te": "tel_Telu",
        "ur": "urd_Arab",
        "ne": "npi_Deva",
        "as": "asm_Beng",
    }

    INDIC_FLORES_CODES = {
        "hin_Deva",
        "mar_Deva",
        "san_Deva",
        "ben_Beng",
        "guj_Gujr",
        "kan_Knda",
        "mal_Mlym",
        "ory_Orya",
        "pan_Guru",
        "tam_Taml",
        "tel_Telu",
        "urd_Arab",
        "npi_Deva",
        "asm_Beng",
    }

    MBART_LANGUAGE_MAP = {
        "en": "en_XX",
        "hi": "hi_IN",
        "sa": "hi_IN",
        "fr": "fr_XX",
        "es": "es_XX",
        "de": "de_DE",
        "it": "it_IT",
        "pt": "pt_XX",
    }

    def __init__(self, model: str = "nllb-600M", device: str = "cpu"):
        self.model_name = model
        self.device = device
        self._model = None
        self._tokenizer = None
        self._google_client = None
        self._model_cache: dict[str, Any] = {}
        self._tokenizer_cache: dict[str, Any] = {}
        self._warned_indictrans_fallback = False
        self._load_model()

    def _load_model(self) -> None:
        if self.model_name == "indictrans2":
            # IndicTrans2 needs different checkpoints for en->indic, indic->en, indic->indic.
            # We load them lazily per translation pair.
            return

        if self.model_name in self.MODEL_IDS:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            hf_model_name = self.MODEL_IDS[self.model_name]
            self._tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(hf_model_name)
            return

        if self.model_name == "google":
            from googletrans import Translator as GoogleTranslator

            self._google_client = GoogleTranslator()

    def _get_or_load_hf_model(self, model_id: str) -> tuple[Any, Any]:
        if model_id not in self._model_cache or model_id not in self._tokenizer_cache:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            is_indictrans = model_id.startswith("ai4bharat/indictrans2")
            self._tokenizer_cache[model_id] = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=is_indictrans,
                use_fast=not is_indictrans,
            )
            self._model_cache[model_id] = AutoModelForSeq2SeqLM.from_pretrained(
                model_id,
                trust_remote_code=is_indictrans,
            )
        return self._model_cache[model_id], self._tokenizer_cache[model_id]

    def _translate_with_nllb_fallback(self, text: str, src_lang: str, tgt_lang: str) -> str:
        fallback_model_id = self.MODEL_IDS["nllb-600M"]
        model, tokenizer = self._get_or_load_hf_model(fallback_model_id)
        fallback_source = self._normalize_nllb_code(src_lang)
        fallback_target = self._normalize_nllb_code(tgt_lang)
        return self._hf_translate_with_model(text, fallback_source, fallback_target, model, tokenizer)

    def _translate_with_model_candidates(self, text: str, source: str, target: str, model_candidates: list[str]) -> str:
        last_error = None
        for model_id in model_candidates:
            try:
                model, tokenizer = self._get_or_load_hf_model(model_id)
                return self._hf_translate_with_model(text, source, target, model, tokenizer)
            except Exception as error:
                last_error = error

        if not self._warned_indictrans_fallback:
            print(
                "IndicTrans2 models are unavailable (likely gated/no-access). "
                "Falling back to NLLB-600M for this run."
            )
            if last_error is not None:
                print(f"Last IndicTrans2 load error: {type(last_error).__name__}: {last_error}")
            self._warned_indictrans_fallback = True

        return self._translate_with_nllb_fallback(text, source, target)

    def _normalize_nllb_code(self, code: str) -> str:
        if not code:
            return "eng_Latn"
        if "_" in code and len(code) >= 8:
            return code
        return self.NLLB_LANGUAGE_MAP.get(code.lower(), "eng_Latn")

    def _normalize_mbart_code(self, code: str) -> str:
        if not code:
            return "en_XX"
        if "_" in code and len(code) >= 4:
            return code
        return self.MBART_LANGUAGE_MAP.get(code.lower(), "en_XX")

    def _normalize_indic_code(self, code: str) -> str:
        if not code:
            return "eng_Latn"
        if "_" in code and len(code) >= 8:
            return code
        normalized = code.lower()
        if normalized in self.INDIC_LANGUAGE_MAP:
            return self.INDIC_LANGUAGE_MAP[normalized]
        # Keep non-Indic languages mappable for fallback routing.
        if normalized in self.NLLB_LANGUAGE_MAP:
            return self.NLLB_LANGUAGE_MAP[normalized]
        return normalized

    def _is_indic_language(self, flores_code: str) -> bool:
        return flores_code in self.INDIC_FLORES_CODES

    def _target_language_token_id(self, target_language_code: str, tokenizer: Any) -> int:
        # Different tokenizer versions expose this differently.
        if hasattr(tokenizer, "lang_code_to_id") and tokenizer.lang_code_to_id:
            if target_language_code in tokenizer.lang_code_to_id:
                return int(tokenizer.lang_code_to_id[target_language_code])

        if hasattr(tokenizer, "get_lang_id"):
            return int(tokenizer.get_lang_id(target_language_code))

        token_id = tokenizer.convert_tokens_to_ids(target_language_code)
        if token_id is None:
            raise ValueError(f"Unsupported target language code: {target_language_code}")
        return int(token_id)

    def _hf_translate_with_model(self, text: str, source_code: str, target_code: str, model: Any, tokenizer: Any) -> str:
        import torch

        is_indictrans = model.__class__.__module__.startswith("transformers_modules.ai4bharat.indictrans2")
        if is_indictrans:
            # IndicTrans2 tokenizer expects: "<src_lang> <tgt_lang> <text>".
            tagged_text = f"{source_code} {target_code} {text}"
            encoded = tokenizer(tagged_text, return_tensors="pt", truncation=True, max_length=512)
            generation_kwargs = {"use_cache": False}
        else:
            tokenizer.src_lang = source_code
            encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            bos_token_id = self._target_language_token_id(target_code, tokenizer)
            generation_kwargs = {"forced_bos_token_id": bos_token_id}

        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_length=512,
                **generation_kwargs,
            )
        return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

    def _hf_translate(self, text: str, source_code: str, target_code: str) -> str:
        return self._hf_translate_with_model(text, source_code, target_code, self._model, self._tokenizer)

    def _indictrans2_translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        source = self._normalize_indic_code(src_lang)
        target = self._normalize_indic_code(tgt_lang)

        if source == target:
            return text

        model_ids = self.MODEL_IDS["indictrans2"]
        source_is_indic = self._is_indic_language(source)
        target_is_indic = self._is_indic_language(target)

        if source == "eng_Latn" and target_is_indic:
            return self._translate_with_model_candidates(text, source, target, model_ids["en_indic"])

        if source_is_indic and target == "eng_Latn":
            return self._translate_with_model_candidates(text, source, target, model_ids["indic_en"])

        if source_is_indic and target_is_indic:
            return self._translate_with_model_candidates(text, source, target, model_ids["indic_indic"])

        # Keep non-Indic language pairs working by falling back to NLLB.
        return self._translate_with_nllb_fallback(text, src_lang, tgt_lang)

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> dict:
        start_time = time.perf_counter()

        if self.model_name == "indictrans2":
            translated_text = self._indictrans2_translate(text, src_lang, tgt_lang)
        elif self.model_name in {"nllb-600M", "nllb-1.3B"}:
            source = self._normalize_nllb_code(src_lang)
            target = self._normalize_nllb_code(tgt_lang)
            translated_text = self._hf_translate(text, source, target)
        elif self.model_name == "mbart50":
            source = self._normalize_mbart_code(src_lang)
            target = self._normalize_mbart_code(tgt_lang)
            translated_text = self._hf_translate(text, source, target)
        elif self.model_name == "opus-mt":
            from transformers import MarianMTModel, MarianTokenizer

            pair = f"{src_lang[:2]}-{tgt_lang[:2]}"
            model_name = f"Helsinki-NLP/opus-mt-{pair}"
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            encoded = tokenizer([text], return_tensors="pt", padding=True)
            translated_text = tokenizer.decode(model.generate(**encoded)[0], skip_special_tokens=True)
        elif self.model_name == "google" and self._google_client is not None:
            translated_text = self._google_client.translate(text, src=src_lang, dest=tgt_lang).text
        else:
            translated_text = text

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
        if self.extractor._vectorizer or self.extractor._ft_model or self.extractor._st_model:
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

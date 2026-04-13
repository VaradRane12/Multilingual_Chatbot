"""
Multilingual Chatbot Pipeline
Stages: Language Detection → Translation to English → Preprocessing
    → Feature Representation → Intent Classification → Reply Generation
    → Machine Translation
"""

import re
import time
import json
import csv
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# STAGE 01: Language Identification
# ─────────────────────────────────────────────

class LanguageDetector:
    """
    Wraps multiple language detection backends so you can swap and compare.
    Supported: 'fasttext', 'langdetect', 'langid'
    """

    def __init__(self, backend: str = "fasttext", model_path: str = None):
        self.backend = backend
        self._load(model_path)

    def _load(self, model_path):
        if self.backend == "fasttext":
            try:
                import fasttext
                path = model_path or "lid.176.ftz"   # download from fastText releases
                self.model = fasttext.load_model(path)
            except Exception as e:
                print(f"[LanguageDetector] fasttext load failed ({e}). Falling back to langdetect.")
                self.backend = "langdetect"
                self._load(None)

        elif self.backend == "langdetect":
            from langdetect import detect, DetectorFactory
            DetectorFactory.seed = 42          # reproducibility
            self._detect_fn = detect

        elif self.backend == "langid":
            import langid
            self._detect_fn = lambda text: langid.classify(text)[0]

        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def detect(self, text: str) -> dict:
        start = time.perf_counter()
        try:
            if self.backend == "fasttext":
                labels, probs = self.model.predict(text.replace("\n", " "), k=1)
                lang = labels[0].replace("__label__", "")
                confidence = float(probs[0])
            else:
                lang = self._detect_fn(text)
                confidence = None          # langdetect/langid don't expose probability easily
        except Exception:
            lang, confidence = "unknown", 0.0

        latency_ms = (time.perf_counter() - start) * 1000
        return {"lang": lang, "confidence": confidence, "latency_ms": round(latency_ms, 2)}


# ─────────────────────────────────────────────
# STAGE 02: Text Preprocessing
# ─────────────────────────────────────────────

class TextPreprocessor:
    """
    Configurable preprocessing:
      tokenizer  : 'whitespace' | 'spacy' | 'sentencepiece'
      stopwords  : 'nltk' | 'none' | 'custom'
      normalize  : 'regex' | 'unicode' | 'full'
    """

    EMOJI_RE  = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        "]+", flags=re.UNICODE)
    URL_RE    = re.compile(r"https?://\S+|www\.\S+")
    MENTION_RE = re.compile(r"@\w+")

    def __init__(self,
                 tokenizer: str = "whitespace",
                 stopwords: str = "nltk",
                 normalize: str = "regex",
                 lang: str = "english",
                 custom_stopwords: list = None):

        self.tokenizer  = tokenizer
        self.stopwords  = stopwords
        self.normalize  = normalize
        self.lang       = lang
        self._stop_set  = self._build_stopset(custom_stopwords)
        self._nlp       = None
        self._sp_model  = None

        if tokenizer == "spacy":
            self._load_spacy()
        elif tokenizer == "sentencepiece":
            self._load_sp()

    def _build_stopset(self, custom):
        if self.stopwords == "none":
            return set()
        if self.stopwords == "custom" and custom:
            return set(custom)
        try:
            from nltk.corpus import stopwords as sw
            import nltk
            nltk.download("stopwords", quiet=True)
            return set(sw.words(self.lang))
        except Exception:
            return set()

    def _load_spacy(self):
        try:
            import spacy
            self._nlp = spacy.load("xx_ent_wiki_sm")   # multilingual
        except Exception:
            print("[Preprocessor] spaCy model not found; falling back to whitespace.")
            self.tokenizer = "whitespace"

    def _load_sp(self):
        try:
            import sentencepiece as spm
            m = spm.SentencePieceProcessor()
            m.Load("spm.model")                        # provide your own .model file
            self._sp_model = m
        except Exception:
            print("[Preprocessor] SentencePiece model not found; falling back to whitespace.")
            self.tokenizer = "whitespace"

    # ── normalization ──────────────────────────

    def _normalize(self, text: str) -> str:
        if self.normalize in ("regex", "full"):
            text = self.URL_RE.sub(" <URL> ", text)
            text = self.EMOJI_RE.sub(" <EMOJI> ", text)
            text = self.MENTION_RE.sub(" <MENTION> ", text)
        if self.normalize in ("unicode", "full"):
            import unicodedata
            text = unicodedata.normalize("NFKC", text)
        if self.normalize == "full":
            text = text.lower()
        return text.strip()

    # ── tokenization ──────────────────────────

    def _tokenize(self, text: str) -> list:
        if self.tokenizer == "spacy" and self._nlp:
            import nltk
            nltk.download("wordnet", quiet=True)
            doc = self._nlp(text)
            return [t.lemma_ for t in doc if not t.is_space]
        if self.tokenizer == "sentencepiece" and self._sp_model:
            return self._sp_model.EncodeAsPieces(text)
        # default: whitespace
        return text.split()

    # ── main entry ─────────────────────────────

    def process(self, text: str) -> dict:
        start = time.perf_counter()
        normalized = self._normalize(text)
        tokens = self._tokenize(normalized)
        tokens = [t for t in tokens if t.lower() not in self._stop_set and len(t) > 1]
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "original": text,
            "normalized": normalized,
            "tokens": tokens,
            "token_count": len(tokens),
            "latency_ms": round(latency_ms, 2),
        }


# ─────────────────────────────────────────────
# STAGE 03: Feature Representation
# ─────────────────────────────────────────────

class FeatureExtractor:
    """
    method: 'tfidf' | 'fasttext_avg' | 'sentence_transformer'
    For sentence_transformer, model_name controls which checkpoint to load.
    """

    def __init__(self, method: str = "tfidf",
                 max_features: int = 20_000,
                 model_name: str = "LaBSE"):
        self.method       = method
        self.max_features = max_features
        self.model_name   = model_name
        self._vectorizer  = None
        self._ft_model    = None
        self._st_model    = None

    # ── fit (call once on training corpus) ────

    def fit(self, corpus: list):
        if self.method == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                sublinear_tf=True,
                ngram_range=(1, 2),
                analyzer="word",
            )
            self._vectorizer.fit(corpus)

        elif self.method == "fasttext_avg":
            import fasttext
            # expects a pre-trained multilingual fasttext model
            self._ft_model = fasttext.load_model("cc.en.300.bin")

        elif self.method == "sentence_transformer":
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(self.model_name)

        return self

    # ── transform ─────────────────────────────

    def transform(self, texts: list):
        import numpy as np
        start = time.perf_counter()

        if self.method == "tfidf":
            if self._vectorizer is None:
                raise RuntimeError("Call fit() before transform().")
            vecs = self._vectorizer.transform(texts)

        elif self.method == "fasttext_avg":
            vecs = np.vstack([
                np.mean([self._ft_model.get_word_vector(w)
                         for w in t.split()] or [np.zeros(300)], axis=0)
                for t in texts
            ])

        elif self.method == "sentence_transformer":
            vecs = self._st_model.encode(texts, show_progress_bar=False)

        latency_ms = (time.perf_counter() - start) * 1000
        return vecs, round(latency_ms, 2)

    def fit_transform(self, corpus: list):
        self.fit(corpus)
        return self.transform(corpus)


# ─────────────────────────────────────────────
# STAGE 04: Intent Classification
# ─────────────────────────────────────────────

class IntentClassifier:
    """
    classifier: 'svm' | 'logreg' | 'naive_bayes' | 'random_forest' | 'mbert'
    """

    CLASSIFIERS = {
        "svm":          ("sklearn.svm",                   "LinearSVC",        {"C": 1.0, "max_iter": 2000}),
        "logreg":       ("sklearn.linear_model",          "LogisticRegression", {"max_iter": 1000, "C": 5.0}),
        "naive_bayes":  ("sklearn.naive_bayes",           "MultinomialNB",    {}),
        "random_forest":("sklearn.ensemble",              "RandomForestClassifier", {"n_estimators": 200, "n_jobs": -1}),
    }

    def __init__(self, classifier: str = "svm"):
        self.classifier = classifier
        self._model     = None
        self._label_enc = None
        self._build()

    def _build(self):
        if self.classifier == "mbert":
            # Placeholder: wire in your fine-tuning loop here
            # e.g. HuggingFace Trainer with bert-base-multilingual-cased
            raise NotImplementedError(
                "mBERT fine-tuning requires a HuggingFace Trainer setup. "
                "See experiments.py run_intent_mbert() for a skeleton."
            )
        module_path, cls_name, kwargs = self.CLASSIFIERS[self.classifier]
        import importlib
        mod = importlib.import_module(module_path)
        self._model = getattr(mod, cls_name)(**kwargs)

    def fit(self, X, y):
        from sklearn.preprocessing import LabelEncoder
        self._label_enc = LabelEncoder()
        y_enc = self._label_enc.fit_transform(y)
        self._model.fit(X, y_enc)
        return self

    def predict(self, X) -> list:
        preds = self._model.predict(X)
        return self._label_enc.inverse_transform(preds).tolist()

    def evaluate(self, X, y_true) -> dict:
        from sklearn.metrics import accuracy_score, f1_score, classification_report
        y_pred = self.predict(X)
        return {
            "accuracy":  round(accuracy_score(y_true, y_pred), 4),
            "macro_f1":  round(f1_score(y_true, y_pred, average="macro"), 4),
            "report":    classification_report(y_true, y_pred),
        }


# ─────────────────────────────────────────────
# STAGE 05: Machine Translation
# ─────────────────────────────────────────────

class Translator:
    """
    model: 'nllb-600M' | 'nllb-1.3B' | 'mbart50' | 'opus-mt' | 'google'
    """

    MODEL_IDS = {
        "nllb-600M": "facebook/nllb-200-distilled-600M",
        "nllb-1.3B": "facebook/nllb-200-1.3B",
        "mbart50":   "facebook/mbart-large-50-many-to-many-mmt",
    }

    def __init__(self, model: str = "nllb-600M", device: str = "cpu"):
        self.model_key = model
        self.device    = device
        self._pipe     = None
        self._load()

    def _load(self):
        if self.model_key in self.MODEL_IDS:
            from transformers import pipeline as hf_pipeline
            self._pipe = hf_pipeline(
                "translation",
                model=self.MODEL_IDS[self.model_key],
                device=0 if self.device == "cuda" else -1,
            )
        elif self.model_key == "opus-mt":
            # Loaded per language pair in translate()
            pass
        elif self.model_key == "google":
            from googletrans import Translator as GTranslator
            self._gtrans = GTranslator()

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> dict:
        """
        NLLB language codes: e.g. 'eng_Latn', 'hin_Deva', 'fra_Latn'
        mBART codes: 'en_XX', 'hi_IN', 'fr_XX'
        """
        start = time.perf_counter()

        if self.model_key in ("nllb-600M", "nllb-1.3B"):
            out = self._pipe(text,
                             src_lang=src_lang,
                             tgt_lang=tgt_lang,
                             max_length=512)
            translation = out[0]["translation_text"]

        elif self.model_key == "mbart50":
            out = self._pipe(text,
                             src_lang=src_lang,
                             tgt_lang=tgt_lang,
                             max_length=512)
            translation = out[0]["translation_text"]

        elif self.model_key == "opus-mt":
            from transformers import MarianMTModel, MarianTokenizer
            pair = f"{src_lang[:2]}-{tgt_lang[:2]}"
            model_name = f"Helsinki-NLP/opus-mt-{pair}"
            tok   = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            ids   = tok([text], return_tensors="pt", padding=True)
            translation = tok.decode(model.generate(**ids)[0],
                                     skip_special_tokens=True)

        elif self.model_key == "google":
            result = self._gtrans.translate(text, src=src_lang, dest=tgt_lang)
            translation = result.text

        else:
            translation = text   # passthrough for unknown

        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "source":      text,
            "translation": translation,
            "src_lang":    src_lang,
            "tgt_lang":    tgt_lang,
            "latency_ms":  round(latency_ms, 2),
        }

    @staticmethod
    def bleu(hypothesis: str, reference: str) -> float:
        """Quick sentence-level BLEU (sacrebleu)."""
        try:
            from sacrebleu.metrics import BLEU
            metric = BLEU(effective_order=True)
            return round(metric.sentence_score(hypothesis, [reference]).score, 2)
        except ImportError:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            import nltk
            nltk.download("punkt", quiet=True)
            ref_tokens  = reference.split()
            hyp_tokens  = hypothesis.split()
            return round(sentence_bleu([ref_tokens], hyp_tokens,
                                       smoothing_function=SmoothingFunction().method1), 4)


# ─────────────────────────────────────────────
# STAGE 06: Reply Generation
# ─────────────────────────────────────────────

class ReplyGenerator:
    """Create a short English reply from the predicted intent."""

    TEMPLATES = {
        "flight": "I can help with flight bookings. Please share the origin, destination, and travel date.",
        "airfare": "I can help with fare information. Which route would you like to check?",
        "quantity": "I can help with quantity details. What exactly do you want to know?",
        "ground_service": "I can help with ground service details. Which airport or city are you asking about?",
        "capacity": "I can help with aircraft capacity. Which flight or aircraft should I check?",
        "aircraft": "I can help with aircraft information. Tell me the airline or flight number.",
        "airline": "I can help with airline details. Which airline are you asking about?",
        "airport": "I can help with airport information. Which city or airport do you want?",
        "meal": "I can help with meal information. Tell me the airline or flight route.",
        "flight_time": "I can help with flight times. Please share the route and travel date.",
        "distance": "I can help with distance information. Which two cities do you want to compare?",
        "restriction": "I can help with travel restrictions. Please share the route or airline.",
        "cheapest": "I can help find cheaper options. Which route are you comparing?",
        "city": "I can help with city-related travel details. Which city do you mean?",
        "abbreviation": "I can help explain abbreviations. Which one do you want me to decode?",
        "flight_no": "I can help with flight numbers. Please share the airline or route.",
        "ground_fare": "I can help with ground fare details. Which city or airport are you asking about?",
        "airline+flight_no": "I can help with airline and flight number details. Please share the route or airline.",
        "aircraft+flight+flight_no": "I can help with aircraft, flight, and flight number details. Please share the route.",
        "airfare+flight_time": "I can help with fare and flight time details. Which route should I check?",
        "flight+airfare": "I can help with flight and fare details. Please share the route and travel date.",
        "ground_service+ground_fare": "I can help with ground service and fare details. Which city or airport are you asking about?",
    }

    def __init__(self, templates_path: str = None):
        self.templates = dict(self.TEMPLATES)
        if templates_path:
            loaded = self._load_templates(templates_path)
            if loaded:
                self.templates = loaded

    def _load_templates(self, templates_path: str) -> dict:
        path = Path(templates_path)
        if not path.exists():
            return {}

        suffix = path.suffix.lower()
        try:
            if suffix == ".json":
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return {str(k): str(v) for k, v in data.items() if str(k).strip() and str(v).strip()}

            if suffix == ".csv":
                templates = {}
                with open(path, "r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        intent = (row.get("intent") or "").strip()
                        reply = (row.get("reply") or "").strip()
                        if intent and reply:
                            templates[intent] = reply
                return templates
        except Exception:
            return {}

        return {}

    def build(self, intent: str) -> str:
        return self.templates.get(
            intent,
            f"I understood this as a {intent.replace('_', ' ')} request. Can you share a bit more detail?",
        )


# ─────────────────────────────────────────────
# Full end-to-end pipeline
# ─────────────────────────────────────────────

class MultilingualChatbotPipeline:
    """
    Chains all five stages. Swap any stage by passing a different config.
    """

    def __init__(self,
                 detector:   LanguageDetector  = None,
                 preprocessor: TextPreprocessor = None,
                 extractor:  FeatureExtractor  = None,
                 classifier: IntentClassifier  = None,
                 translator: Translator        = None,
                 reply_templates_path: str     = None):

        self.detector     = detector     or LanguageDetector(backend="langdetect")
        self.preprocessor = preprocessor or TextPreprocessor()
        self.extractor    = extractor    or FeatureExtractor(method="tfidf")
        self.classifier   = classifier   # optional — set after fit()
        self.translator   = translator   # optional — heavy, load on demand
        self.reply_generator = ReplyGenerator(reply_templates_path)

    def _build_reply(self, intent: str, src_lang: str, tgt_lang: str | None) -> dict:
        reply_english = self.reply_generator.build(intent)
        reply_text = reply_english
        reply_translation = None

        if self.translator and tgt_lang and tgt_lang != "en":
            reply_translation = self.translator.translate(reply_english, "en", tgt_lang)
            reply_text = reply_translation["translation"]

        return {
            "reply_english": reply_english,
            "reply": reply_text,
            "reply_translation": reply_translation,
            "reply_language": tgt_lang or src_lang,
        }

    def run(self, text: str, tgt_lang: str = None) -> dict:
        result = {}

        # Stage 1: Detect input language
        result["detection"] = self.detector.detect(text)
        src_lang = result["detection"]["lang"]
        reply_target_lang = tgt_lang or src_lang

        # Stage 2: Translate to English if input is not English (needed for ML pipeline)
        pipeline_text = text  # text to use for feature extraction
        if src_lang != "en" and src_lang != "unknown":
            if self.translator:
                trans_result = self.translator.translate(text, src_lang, "en")
                pipeline_text = trans_result["translation"]
                result["input_translation"] = trans_result
        
        # Stage 3: Preprocess (now on English text)
        result["preprocessing"] = self.preprocessor.process(pipeline_text)
        clean_text = " ".join(result["preprocessing"]["tokens"])

        # Stage 4: Feature extraction (inference only — assumes extractor already fitted)
        if self.extractor._vectorizer or self.extractor._st_model or self.extractor._ft_model:
            vecs, lat = self.extractor.transform([clean_text])
            result["features"] = {"shape": list(vecs.shape) if hasattr(vecs, "shape") else "sparse",
                                   "latency_ms": lat}

        # Stage 5: Classification (on English features)
        if self.classifier:
            vecs, _ = self.extractor.transform([clean_text])
            result["intent"] = self.classifier.predict(vecs)[0]
            result.update(self._build_reply(result["intent"], src_lang, reply_target_lang))

        # Stage 6: keep translation metadata for the UI
        if self.translator and reply_target_lang:
            result["translation"] = self.translator.translate(text, src_lang, reply_target_lang)

        return result


if __name__ == "__main__":
    print("pipeline.py defines reusable pipeline classes.")
    print("To run the interactive app, use:")
    print("  streamlit run streamlit_app.py")

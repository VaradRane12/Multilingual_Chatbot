"""
experiments.py — Compare all pipeline configurations and print a results table.

Run:
    python experiments.py

Each experiment function is independent — comment out stages you don't need
while you're still setting up dependencies.
"""

import numpy as np

from dataset_utils import load_config, load_intent_dataset

# ── helpers ────────────────────────────────────────────────────────────────────

def print_header(title: str):
    print("\n" + "═" * 60)
    print(f"  {title}")
    print("═" * 60)

# ── EXPERIMENT 1: Language Detection ───────────────────────────────────────────

def experiment_language_detection():
    print_header("EXPERIMENT 01 — Language Detection")

    # Short / medium / long samples in various languages
    samples = {
        "short_en":  "Book a flight",
        "short_hi":  "मुझे टिकट चाहिए",
        "short_fr":  "Réserver un vol",
        "medium_en": "I need to book a flight from Mumbai to Delhi for next Monday.",
        "medium_es": "Necesito reservar un vuelo de Madrid a Barcelona para el lunes.",
        "long_de":   ("Ich möchte einen Flug von Frankfurt nach Berlin buchen und "
                      "benötige außerdem Informationen über die Gepäckbestimmungen."),
    }

    backends = ["langdetect", "langid"]
    # Add "fasttext" here if you have lid.176.ftz downloaded

    from pipeline import LanguageDetector
    results = {}

    for backend in backends:
        try:
            det = LanguageDetector(backend=backend)
            backend_results = {}
            for name, text in samples.items():
                res = det.detect(text)
                backend_results[name] = res
            results[backend] = backend_results
            print(f"\n  Backend: {backend}")
            for name, r in backend_results.items():
                conf = f"{r['confidence']:.3f}" if r["confidence"] else "n/a "
                print(f"    {name:<15} → lang={r['lang']:<6}  conf={conf}  latency={r['latency_ms']}ms")
        except Exception as e:
            print(f"  [{backend}] skipped: {e}")

    return results


# ── EXPERIMENT 2: Preprocessing Configurations ─────────────────────────────────

def experiment_preprocessing():
    print_header("EXPERIMENT 02 — Preprocessing Configs")

    texts = [
        "Hello! Check out https://example.com 😊 it's amazing",
        "मुझे अभी एक टिकट बुक करनी है @support #help",
        "Hola, ¿cómo reservo un vuelo? #viaje www.airline.es",
    ]

    configs = [
        ("baseline", dict(tokenizer="whitespace", stopwords="nltk", normalize="regex"), None),
        ("no-stopwords", dict(tokenizer="whitespace", stopwords="none", normalize="regex"), None),
        ("full-normalize", dict(tokenizer="whitespace", stopwords="nltk", normalize="full"), None),
        (
            "custom-stop",
            dict(tokenizer="whitespace", stopwords="custom", normalize="regex"),
            ["the", "a", "an", "is", "it", "in", "on", "at", "to", "for"],
        ),
    ]

    from pipeline import TextPreprocessor

    for label, cfg, custom in configs:
        pp = TextPreprocessor(**cfg, custom_stopwords=custom)
        print(f"\n  Config: {label}")
        for text in texts[:2]:           # first 2 to keep output compact
            res = pp.process(text)
            print(f"    in : {text[:50]}")
            print(f"    out: {' '.join(res['tokens'][:10])}  ({res['token_count']} tokens, {res['latency_ms']}ms)")


# ── EXPERIMENT 3: Feature Representation ──────────────────────────────────────

def experiment_features():
    print_header("EXPERIMENT 03 — Feature Representation")

    config = load_config("config.json")
    data = load_intent_dataset(config)
    corpus = data["train_texts"]
    targets = np.array(data["train_intents"])

    from pipeline import FeatureExtractor, IntentClassifier
    from sklearn.model_selection import cross_val_score

    configs = [
        ("TF-IDF 5k", dict(method="tfidf", max_features=5_000)),
        ("TF-IDF 20k", dict(method="tfidf", max_features=20_000)),
        ("TF-IDF 50k", dict(method="tfidf", max_features=50_000)),
        # Uncomment when sentence-transformers is installed:
        # ("LaBSE", dict(method="sentence_transformer", model_name="LaBSE")),
        # ("mE5-large", dict(method="sentence_transformer", model_name="intfloat/multilingual-e5-large")),
    ]

    for label, cfg in configs:
        fe = FeatureExtractor(**cfg)
        try:
            vecs, lat = fe.fit_transform(corpus)
            # Quick cross-val with SVM
            from sklearn.svm import LinearSVC
            scores = cross_val_score(LinearSVC(max_iter=2000), vecs, targets,
                                     cv=3, scoring="accuracy")
            print(f"\n  {label:<20} shape={vecs.shape}  "
                  f"cv_acc={scores.mean():.3f}±{scores.std():.3f}  fit+transform={lat:.1f}ms")
        except Exception as e:
            print(f"\n  {label} — skipped: {e}")


# ── EXPERIMENT 4: Intent Classification ───────────────────────────────────────

def experiment_intent():
    print_header("EXPERIMENT 04 — Intent Classification")

    config = load_config("config.json")
    data = load_intent_dataset(config)
    X_train = data["train_texts"]
    y_train = data["train_intents"]
    X_test = data["test_texts"] or data["train_texts"]
    y_test = data["test_intents"] or data["train_intents"]

    from pipeline import FeatureExtractor, IntentClassifier

    fe = FeatureExtractor(method="tfidf", max_features=5_000)
    fe.fit(X_train)
    X_tr_vec, _ = fe.transform(X_train)
    X_te_vec, _ = fe.transform(X_test)

    classifiers = ["logreg", "svm", "naive_bayes", "random_forest"]

    print(f"\n  {'Classifier':<20} {'Accuracy':>10} {'Macro F1':>10}")
    print("  " + "-" * 42)

    best_clf = None
    best_acc = 0

    for clf_name in classifiers:
        try:
            clf = IntentClassifier(classifier=clf_name)
            clf.fit(X_tr_vec, y_train)
            metrics = clf.evaluate(X_te_vec, y_test)
            acc = metrics["accuracy"]
            f1  = metrics["macro_f1"]
            print(f"  {clf_name:<20} {acc:>10.4f} {f1:>10.4f}")
            if acc > best_acc:
                best_acc = acc
                best_clf = clf_name
        except Exception as e:
            print(f"  {clf_name:<20} skipped: {e}")

    print(f"\n  → Best classifier: {best_clf} (acc={best_acc:.4f})")
    return best_clf


# ── EXPERIMENT 5: Machine Translation ─────────────────────────────────────────

def experiment_translation():
    print_header("EXPERIMENT 05 — Machine Translation")

    # Reference pairs: (source_text, reference_translation, src_lang, tgt_lang)
    # NLLB language codes used here
    test_pairs = [
        ("Book a flight to Paris",
         "Réserver un vol pour Paris",
         "eng_Latn", "fra_Latn"),
        ("Cancel my booking please",
         "Por favor cancela mi reserva",
         "eng_Latn", "spa_Latn"),
        ("What is the baggage allowance?",
         "क्या है बैगेज अलाउंस?",
         "eng_Latn", "hin_Deva"),
    ]

    models_to_test = [
        "nllb-600M",
        # "nllb-1.3B",    # uncomment if you have RAM / GPU
        # "opus-mt",
    ]

    from pipeline import Translator

    for model_key in models_to_test:
        print(f"\n  Model: {model_key}")
        try:
            tr = Translator(model=model_key)
            bleu_scores = []
            for src, ref, src_l, tgt_l in test_pairs:
                res  = tr.translate(src, src_l, tgt_l)
                bleu = Translator.bleu(res["translation"], ref)
                bleu_scores.append(bleu)
                print(f"    src : {src}")
                print(f"    pred: {res['translation']}")
                print(f"    ref : {ref}")
                print(f"    BLEU: {bleu}  latency: {res['latency_ms']}ms\n")
            print(f"  Avg BLEU: {np.mean(bleu_scores):.2f}")
        except Exception as e:
            print(f"  [{model_key}] skipped: {e}")


# ── EXPERIMENT 6: End-to-End Pipeline ─────────────────────────────────────────

def experiment_end_to_end():
    print_header("EXPERIMENT 06 — End-to-End Pipeline (no translation)")

    from pipeline import (MultilingualChatbotPipeline,
                           LanguageDetector, TextPreprocessor,
                           FeatureExtractor, IntentClassifier)

    config = load_config("config.json")
    data = load_intent_dataset(config)
    train_texts = data["train_texts"]
    train_labels = data["train_intents"]
    test_texts = data["test_texts"]

    fe  = FeatureExtractor(method="tfidf", max_features=2_000)
    fe.fit(train_texts)
    X, _ = fe.transform(train_texts)

    clf = IntentClassifier(classifier="svm")
    clf.fit(X, train_labels)

    pipeline = MultilingualChatbotPipeline(
        detector     = LanguageDetector(backend="langdetect"),
        preprocessor = TextPreprocessor(tokenizer="whitespace",
                                        stopwords="nltk",
                                        normalize="regex"),
        extractor    = fe,
        classifier   = clf,
    )

    queries = test_texts[:4] if test_texts else train_texts[:4]

    print()
    for q in queries:
        res = pipeline.run(q)
        lang   = res["detection"]["lang"]
        intent = res.get("intent", "n/a")
        tokens = res["preprocessing"]["token_count"]
        print(f"  Query  : {q}")
        print(f"  Lang   : {lang}  |  Tokens: {tokens}  |  Intent: {intent}")
        print()


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🔬 Multilingual Chatbot Pipeline — Experiments")
    print("=" * 60)

    experiment_language_detection()
    experiment_preprocessing()
    experiment_features()
    best = experiment_intent()
    experiment_end_to_end()

    # Heavy — comment in when HuggingFace models are downloaded:
    # experiment_translation()

    print("\n✅ All experiments complete.\n")

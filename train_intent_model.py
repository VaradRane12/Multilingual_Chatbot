"""Train an intent classifier from configured dataset and save artifacts.

Usage:
    python train_intent_model.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from dataset_utils import load_config, load_intent_dataset
from pipeline import FeatureExtractor, IntentClassifier


def main() -> None:
    config = load_config("config.json")
    data = load_intent_dataset(config)

    train_texts = data["train_texts"]
    train_intents = data["train_intents"]
    test_texts = data["test_texts"]
    test_intents = data["test_intents"]

    if not train_texts or not train_intents:
        raise RuntimeError("No training samples found after dataset load/cleanup.")

    model_cfg = config["model"]
    candidate_classifiers = model_cfg.get(
        "candidate_classifiers",
        ["svm", "logreg", "naive_bayes", "random_forest"],
    )
    extractor = FeatureExtractor(method="tfidf", max_features=int(model_cfg.get("max_features", 20000)))
    extractor.fit(train_texts)
    X_train, _ = extractor.transform(train_texts)

    X_test = None
    test_metrics = []
    if test_texts and test_intents:
        X_test, _ = extractor.transform(test_texts)

    classifiers_dir = Path(config["runtime"].get("artifacts_dir", "artifacts")) / "classifiers"
    classifiers_dir.mkdir(parents=True, exist_ok=True)

    comparison = []
    saved_paths = {}
    best_classifier_name = None
    best_score = -1.0

    for classifier_name in candidate_classifiers:
        classifier = IntentClassifier(classifier=classifier_name)
        classifier.fit(X_train, train_intents)

        metrics = {}
        if X_test is not None:
            metrics = classifier.evaluate(X_test, test_intents)

        artifact_path = classifiers_dir / f"{classifier_name}.joblib"
        joblib.dump(classifier, artifact_path)
        saved_paths[classifier_name] = str(artifact_path)

        row = {"classifier": classifier_name, **metrics}
        comparison.append(row)

        score = float(metrics.get("accuracy", 0.0)) if metrics else 0.0
        if score > best_score:
            best_score = score
            best_classifier_name = classifier_name

    runtime_cfg = config["runtime"]
    artifacts_dir = Path(runtime_cfg.get("artifacts_dir", "artifacts"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(extractor, artifacts_dir / "extractor.joblib")
    if best_classifier_name is not None:
        joblib.dump(joblib.load(saved_paths[best_classifier_name]), artifacts_dir / "classifier.joblib")

    metadata = {
        "dataset": config["dataset"],
        "model": model_cfg,
        "sample_count": len(train_texts),
        "intent_count": len(set(train_intents)),
        "best_classifier": best_classifier_name,
        "comparison": comparison,
        "classifier_artifacts": saved_paths,
    }
    with open(artifacts_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("Training complete.")
    print(f"Saved artifacts to: {artifacts_dir}")
    if comparison:
        for row in comparison:
            print(
                f"{row['classifier']}: accuracy={row.get('accuracy')} macro_f1={row.get('macro_f1')}"
            )
        print(f"Best classifier: {best_classifier_name}")


if __name__ == "__main__":
    main()

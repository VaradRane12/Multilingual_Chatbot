"""Create graphs for translation BLEU and classifier performance.

Usage:
    python make_graphs.py
    python make_graphs.py --translation scored_translation_eval_all.csv --outdir charts
    python make_graphs.py --translation 200m.csv --translation 600m.csv --metadata artifacts/metadata.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def safe_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def plot_translation_language_bleu(rows: list[dict[str, str]], title: str, output_path: Path) -> None:
    scores_by_language = defaultdict(list)
    for row in rows:
        target_lang = row.get("target_lang", "unknown")
        bleu = safe_float(row.get("bleu", "0"))
        scores_by_language[target_lang].append(bleu)

    languages = sorted(scores_by_language.keys())
    averages = [sum(scores_by_language[lang]) / len(scores_by_language[lang]) for lang in languages]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(languages, averages, color="#4C78A8")
    plt.title(title)
    plt.xlabel("Target language")
    plt.ylabel("Average BLEU")
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, averages):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{value:.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_translation_latency(rows: list[dict[str, str]], title: str, output_path: Path) -> None:
    scores_by_language = defaultdict(list)
    for row in rows:
        target_lang = row.get("target_lang", "unknown")
        latency = safe_float(row.get("latency_ms", "0"))
        scores_by_language[target_lang].append(latency)

    languages = sorted(scores_by_language.keys())
    averages = [sum(scores_by_language[lang]) / len(scores_by_language[lang]) for lang in languages]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(languages, averages, color="#F58518")
    plt.title(title)
    plt.xlabel("Target language")
    plt.ylabel("Average latency (ms)")
    plt.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, averages):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(averages) * 0.01, f"{value:.0f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_translation_model_comparison(model_results: dict[str, list[dict[str, str]]], output_path: Path) -> None:
    model_names = []
    avg_bleu_values = []

    for model_name, rows in model_results.items():
        bleu_values = [safe_float(row.get("bleu", "0")) for row in rows]
        if not bleu_values:
            continue
        model_names.append(model_name)
        avg_bleu_values.append(sum(bleu_values) / len(bleu_values))

    if not model_names:
        return

    plt.figure(figsize=(8, 5))
    bars = plt.bar(model_names, avg_bleu_values, color="#54A24B")
    plt.title("Translation model comparison")
    plt.xlabel("Model")
    plt.ylabel("Average BLEU")
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, avg_bleu_values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{value:.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_classifier_comparison(metadata: dict, output_path: Path) -> None:
    comparison = metadata.get("comparison", [])
    if not comparison:
        return

    classifiers = [row.get("classifier", "unknown") for row in comparison]
    accuracies = [safe_float(str(row.get("accuracy", 0.0))) * 100.0 for row in comparison]
    macro_f1s = [safe_float(str(row.get("macro_f1", 0.0))) * 100.0 for row in comparison]

    x = range(len(classifiers))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar([pos - width / 2 for pos in x], accuracies, width=width, label="Accuracy", color="#4C78A8")
    plt.bar([pos + width / 2 for pos in x], macro_f1s, width=width, label="Macro F1", color="#E45756")
    plt.xticks(list(x), classifiers)
    plt.title("Intent classifier comparison")
    plt.xlabel("Classifier")
    plt.ylabel("Score (%)")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_classifier_ranking(metadata: dict, output_path: Path) -> None:
    comparison = metadata.get("comparison", [])
    if not comparison:
        return

    sorted_rows = sorted(comparison, key=lambda row: safe_float(str(row.get("accuracy", 0.0))), reverse=True)
    classifiers = [row.get("classifier", "unknown") for row in sorted_rows]
    accuracies = [safe_float(str(row.get("accuracy", 0.0))) * 100.0 for row in sorted_rows]

    plt.figure(figsize=(10, 5))
    bars = plt.barh(classifiers, accuracies, color="#72B7B2")
    plt.title("Classifier ranking by accuracy")
    plt.xlabel("Accuracy (%)")
    plt.xlim(0, 100)
    plt.grid(axis="x", alpha=0.25)

    for bar, value in zip(bars, accuracies):
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, f"{value:.1f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create graphs from translation and classifier evaluation files.")
    parser.add_argument("--translation", action="append", default=[], help="Path to a scored translation CSV. You can pass this multiple times.")
    parser.add_argument("--metadata", default="artifacts/metadata.json", help="Path to model metadata JSON.")
    parser.add_argument("--outdir", default="charts", help="Directory to save graphs.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    ensure_outdir(outdir)

    translation_files = [Path(path) for path in args.translation]
    model_results: dict[str, list[dict[str, str]]] = {}

    for translation_file in translation_files:
        if not translation_file.exists():
            raise FileNotFoundError(f"Translation file not found: {translation_file}")
        model_name = translation_file.stem
        model_results[model_name] = read_csv_rows(translation_file)

        plot_translation_language_bleu(
            model_results[model_name],
            f"Average BLEU by language - {model_name}",
            outdir / f"{model_name}_bleu_by_language.png",
        )
        plot_translation_latency(
            model_results[model_name],
            f"Average latency by language - {model_name}",
            outdir / f"{model_name}_latency_by_language.png",
        )

    if len(model_results) > 1:
        plot_translation_model_comparison(model_results, outdir / "translation_model_comparison.png")

    metadata_path = Path(args.metadata)
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)
        plot_classifier_comparison(metadata, outdir / "classifier_comparison.png")
        plot_classifier_ranking(metadata, outdir / "classifier_ranking.png")

    print(f"Saved graphs to: {outdir}")
    if translation_files:
        print("Translation graphs created for:")
        for translation_file in translation_files:
            print(f"  - {translation_file}")
    if metadata_path.exists():
        print(f"Classifier graphs created from: {metadata_path}")


if __name__ == "__main__":
    main()

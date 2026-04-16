"""Evaluate translation quality with BLEU.

Usage:
    python evaluate_translation.py
    python evaluate_translation.py --input translation_eval.csv --model nllb-600M
    python evaluate_translation.py --output scored_translation_eval.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from pipeline import Translator


def compute_bleu_score(candidate_text: str, reference_text: str) -> float:
    # Use sacrebleu when available, otherwise fall back to a tiny built-in BLEU.
    try:
        import importlib

        sacrebleu_metrics = importlib.import_module("sacrebleu.metrics")
        metric = sacrebleu_metrics.BLEU(effective_order=True)
        return round(float(metric.sentence_score(candidate_text, [reference_text]).score), 2)
    except Exception:
        reference_tokens = reference_text.split()
        candidate_tokens = candidate_text.split()
        if not reference_tokens or not candidate_tokens:
            return 0.0

        def ngram_counts(tokens: list[str], n: int) -> dict[tuple[str, ...], int]:
            counts: dict[tuple[str, ...], int] = {}
            for i in range(len(tokens) - n + 1):
                key = tuple(tokens[i : i + n])
                counts[key] = counts.get(key, 0) + 1
            return counts

        precisions: list[float] = []
        for n in (1, 2, 3, 4):
            candidate_ngrams = ngram_counts(candidate_tokens, n)
            reference_ngrams = ngram_counts(reference_tokens, n)
            overlap = 0
            total = 0
            for ngram, count in candidate_ngrams.items():
                overlap += min(count, reference_ngrams.get(ngram, 0))
                total += count
            precisions.append((overlap + 1.0) / (total + 1.0))

        geometric_mean = math.exp(sum(math.log(score) for score in precisions) / 4.0)
        candidate_len = len(candidate_tokens)
        reference_len = len(reference_tokens)
        brevity_penalty = 1.0 if candidate_len > reference_len else math.exp(1.0 - (reference_len / max(candidate_len, 1)))
        return round(float(100.0 * brevity_penalty * geometric_mean), 2)


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate translation quality with BLEU.")
    parser.add_argument("--input", default="translation_eval.csv", help="CSV file with source_text, source_lang, target_lang, reference_translation")
    parser.add_argument("--model", default="nllb-600M", help="Translator model name")
    parser.add_argument("--output", default="", help="Optional CSV path to save scored results")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    rows = read_rows(input_path)
    if not rows:
        raise RuntimeError("No rows found in the input CSV.")

    translator = Translator(model=args.model)
    scored_rows: list[dict[str, str]] = []
    bleu_scores: list[float] = []
    bleu_by_language: dict[str, list[float]] = defaultdict(list)

    print(f"Evaluating {len(rows)} rows with model: {args.model}")
    print("-" * 80)

    for index, row in enumerate(rows, start=1):
        source_text = row.get("source_text", "").strip()
        source_lang = row.get("source_lang", "").strip()
        target_lang = row.get("target_lang", "").strip()
        reference_translation = row.get("reference_translation", "").strip()

        if not source_text or not source_lang or not target_lang or not reference_translation:
            print(f"Row {index}: skipped because a required field is empty")
            continue

        translation_result = translator.translate(source_text, source_lang, target_lang)
        predicted_translation = translation_result.get("translation", "")
        bleu_score = compute_bleu_score(predicted_translation, reference_translation)
        bleu_scores.append(bleu_score)
        bleu_by_language[target_lang].append(bleu_score)

        scored_row = {
            **row,
            "predicted_translation": predicted_translation,
            "bleu": f"{bleu_score:.2f}",
            "latency_ms": f"{translation_result.get('latency_ms', 0.0):.2f}",
        }
        scored_rows.append(scored_row)

        print(f"Row {index}")
        print(f"  Source:    {source_text}")
        print(f"  Predicted: {predicted_translation}")
        print(f"  Reference: {reference_translation}")
        print(f"  BLEU:      {bleu_score:.2f}")
        print(f"  Latency:   {translation_result.get('latency_ms', 0.0):.2f} ms")
        print()

    if bleu_scores:
        average_bleu = sum(bleu_scores) / len(bleu_scores)
        print("-" * 80)
        print(f"Average BLEU: {average_bleu:.2f}")

        print("BLEU by target language:")
        for language_code in sorted(bleu_by_language.keys()):
            language_scores = bleu_by_language[language_code]
            language_average = sum(language_scores) / len(language_scores)
            print(f"  {language_code}: {language_average:.2f} ({len(language_scores)} rows)")
    else:
        print("No valid rows were scored.")

    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["source_text", "source_lang", "target_lang", "reference_translation", "predicted_translation", "bleu", "latency_ms"])
            writer.writeheader()
            writer.writerows(scored_rows)
        print(f"Saved scored rows to: {output_path}")


if __name__ == "__main__":
    main()

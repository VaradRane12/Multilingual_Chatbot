"""Dataset and configuration helpers for the intent pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def load_config(config_path: str = "config.json") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _clean_pairs(texts: List[str], intents: List[str]) -> Dict[str, List[str]]:
    clean_texts = []
    clean_intents = []
    for text, intent in zip(texts, intents):
        if text is None or intent is None:
            continue
        text_s = str(text).strip()
        intent_s = str(intent).strip()
        if not text_s or not intent_s:
            continue
        clean_texts.append(text_s)
        clean_intents.append(intent_s)
    return {"texts": clean_texts, "intents": clean_intents}


def load_intent_dataset(config: dict) -> dict:
    dataset_cfg = config["dataset"]
    source = dataset_cfg.get("source", "huggingface")

    if source == "huggingface":
        from datasets import load_dataset

        dataset_id = dataset_cfg["dataset_id"]
        train_split = dataset_cfg.get("train_split", "train")
        test_split = dataset_cfg.get("test_split", "test")
        text_col = dataset_cfg.get("text_column", "text")
        intent_col = dataset_cfg.get("intent_column", "intent")

        train_ds = load_dataset(dataset_id, split=train_split)
        train_clean = _clean_pairs(train_ds[text_col], train_ds[intent_col])

        test_ds = load_dataset(dataset_id, split=test_split)
        test_clean = _clean_pairs(test_ds[text_col], test_ds[intent_col])
        test_texts = test_clean["texts"]
        test_intents = test_clean["intents"]

        return {
            "train_texts": train_clean["texts"],
            "train_intents": train_clean["intents"],
            "test_texts": test_texts,
            "test_intents": test_intents,
        }

    if source == "csv":
        import pandas as pd
        from sklearn.model_selection import train_test_split

        csv_path = Path(dataset_cfg.get("csv_path", "data/intent_dataset.csv"))
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV dataset not found at {csv_path}")

        text_col = dataset_cfg.get("text_column", "text")
        intent_col = dataset_cfg.get("intent_column", "intent")
        split_col = dataset_cfg.get("split_column", "split")

        df = pd.read_csv(csv_path)
        if text_col not in df.columns or intent_col not in df.columns:
            raise ValueError(
                f"CSV must contain '{text_col}' and '{intent_col}' columns. Found: {list(df.columns)}"
            )

        if split_col in df.columns:
            train_df = df[df[split_col].astype(str).str.lower() == "train"]
            test_df = df[df[split_col].astype(str).str.lower() == "test"]
        else:
            train_df, test_df = train_test_split(
                df,
                train_size=dataset_cfg.get("train_ratio", 0.8),
                random_state=dataset_cfg.get("random_state", 42),
                stratify=df[intent_col],
            )

        train_clean = _clean_pairs(train_df[text_col].tolist(), train_df[intent_col].tolist())
        test_clean = _clean_pairs(test_df[text_col].tolist(), test_df[intent_col].tolist())

        return {
            "train_texts": train_clean["texts"],
            "train_intents": train_clean["intents"],
            "test_texts": test_clean["texts"],
            "test_intents": test_clean["intents"],
        }

    raise ValueError(f"Unsupported dataset source: {source}")

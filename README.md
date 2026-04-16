# Multilingual Chatbot

This project does one main thing:
it predicts travel intent (like flight, airfare, airport) from user text,
and it can reply in the same language as the user.

## Main files

- `train_intent_model.py`: trains models and saves artifacts.
- `streamlit_app.py`: chat UI.
- `pipeline.py`: core detection, preprocessing, features, classifier, translation.
- `dataset_utils.py`: reads config and dataset.
- `config.json`: all settings.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train models:

```bash
python train_intent_model.py
```

3. Run app:

```bash
streamlit run streamlit_app.py
```

## Notes

- Trained files are saved in `artifacts/`.
- App reads model comparison and best model from `artifacts/metadata.json`.
- Translator is set to `nllb-600M`.

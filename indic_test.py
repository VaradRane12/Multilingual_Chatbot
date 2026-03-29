# Use a pipeline as a high-level helper
# Warning: Pipeline type "translation" is no longer supported in transformers v5.
# You must load the model directly (see below) or downgrade to v4.x with:
# 'pip install "transformers<5.0.0'
from transformers import pipeline

pipe = pipeline("translation", model="ai4bharat/indictrans2-en-indic-dist-200M", trust_remote_code=True)
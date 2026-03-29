import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# Load data
df = pd.read_csv("intents.csv")

X = df["text"]
y = df["intent"]

# TF-IDF
vectorizer = TfidfVectorizer(ngram_range=(1, 1), stop_words="english")
X_vec = vectorizer.fit_transform(X)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_vec, y)

# Save
joblib.dump(model, "intent_model.joblib")
joblib.dump(vectorizer, "tfidf_vectorizer.joblib")

print("Intent model trained and saved.")

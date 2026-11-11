"""
Train the sentiment model locally and save it as model.pkl.

    python train_test.py

app.py uses model.pkl if it exists; otherwise it trains the model itself.
"""
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

df = pd.read_csv("IMDB Dataset.csv")
X_train, X_test, y_train, y_test = train_test_split(
    df[["review"]], df["sentiment"], test_size=0.2, random_state=42
)

model = Pipeline([
    ("features", ColumnTransformer([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english"), "review"),
    ])),
    ("clf", LogisticRegression(max_iter=1000)),
])
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("F1 Score:", round(f1_score(y_test, y_pred, pos_label="positive"), 4))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Saved: model.pkl")

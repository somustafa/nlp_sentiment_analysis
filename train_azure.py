"""
IMDB rəyləri üzrə sentiment modeli — Azure Machine Learning üçün.

İşə salmaq (Azure ML notebook terminalında):
    python train_azure.py --data "<Data asset-in URI-si>"

Nə edir:
  1. Dataset-i bulud yaddaşından oxuyur
  2. TF-IDF + Logistic Regression modelini öyrədir
  3. Accuracy və F1-i Azure ML-ə yazır
  4. Modeli "imdb-sentiment" adı ilə qeydiyyata alır (sonra bir kliklə deploy olunur)
"""
import argparse

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="IMDB Dataset.csv", help="CSV faylının yolu və ya Azure URI-si")
args = parser.parse_args()

# 1. Verilənləri oxu
df = pd.read_csv(args.data)
X = df[["review"]]          # model "review" sütunu olan cədvəl qəbul edir
y = df["sentiment"]         # "positive" / "negative"

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Model: mətn -> TF-IDF -> Logistic Regression (hamısı bir pipeline-da)
model = Pipeline([
    ("features", ColumnTransformer([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english"), "review"),
    ])),
    ("clf", LogisticRegression(max_iter=1000)),
])

with mlflow.start_run():
    model.fit(X_train, y_train)

    # 3. Qiymətləndirmə
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label="positive")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1", f1)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # 4. Modeli qeydiyyata al
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="imdb-sentiment",
        signature=infer_signature(X_test, y_pred),
        input_example=X_test.head(2),
    )
    print("Model 'imdb-sentiment' adı ilə qeydiyyata alındı.")

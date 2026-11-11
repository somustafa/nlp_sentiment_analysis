"""
IMDB Sentiment Analysis — Streamlit app.

Runs locally, on an Azure ML compute instance and on Streamlit Community Cloud.

Model loading order:
  1. Azure ML Model Registry ("imdb-sentiment")  — only on Azure
  2. model.pkl                                   — if you ran train_test.py
  3. Train from "IMDB Dataset.csv" on first start (then cached)
"""
import os
import pickle

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATA_FILE = "IMDB Dataset.csv"
MODEL_FILE = "model.pkl"
REGISTRY_MODEL = "imdb-sentiment"

st.set_page_config(page_title="IMDB Sentiment Analysis", layout="centered")

st.markdown(
    """
    <style>
      #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {display: none;}
      [data-testid="stHeader"] {background: transparent;}
      .block-container {max-width: 720px; padding-top: 4rem;}
      .app-title {font-size: 2.2rem !important; font-weight: 700; letter-spacing: -0.02em; line-height: 1.2; margin: 0;}
      .app-subtitle {opacity: 0.65; margin: 0.5rem 0 1.75rem 0; font-size: 1.05rem !important;}
      .stTextArea textarea {font-size: 1rem; line-height: 1.5;}
      [data-testid="stElementContainer"]:has(.stButton), .stButton {width: 100% !important;}
      .stButton button {width: 100% !important; padding: 0.65rem 1rem !important; border-radius: 8px !important;}
      .stButton button p {font-weight: 600; font-size: 1rem;}
      .stButton button[kind="primary"] {background: #2563eb; border-color: #2563eb;}
      .stButton button[kind="primary"]:hover {background: #1d4ed8; border-color: #1d4ed8;}
      .result {margin-top: 1.5rem; padding: 1.25rem 1.5rem; border-radius: 10px; border-left: 6px solid;}
      .result.positive {background: rgba(22, 163, 74, 0.10); border-color: #16a34a;}
      .result.negative {background: rgba(220, 38, 38, 0.10); border-color: #dc2626;}
      .result-label {font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; opacity: 0.7;}
      .result-value {font-size: 1.6rem; font-weight: 700; margin-top: 0.2rem;}
      .result.positive .result-value {color: #16a34a;}
      .result.negative .result-value {color: #dc2626;}
      .result-conf {margin-top: 0.4rem; opacity: 0.75;}
    </style>
    """,
    unsafe_allow_html=True,
)


def on_azure():
    return bool(os.getenv("CI_NAME")) or os.path.exists("/mnt/batch/tasks/shared/LS_root")


def build_pipeline():
    return Pipeline([
        ("features", ColumnTransformer([
            ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english"), "review"),
        ])),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


@st.cache_resource(show_spinner="Model yüklənir...")
def load_model():
    if on_azure():
        try:
            import mlflow

            client = mlflow.tracking.MlflowClient()
            versions = client.search_model_versions(f"name='{REGISTRY_MODEL}'")
            latest = max(int(v.version) for v in versions)
            return mlflow.sklearn.load_model(f"models:/{REGISTRY_MODEL}/{latest}")
        except Exception:
            pass

    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            return pickle.load(f)

    df = pd.read_csv(DATA_FILE)
    X_train, _, y_train, _ = train_test_split(
        df[["review"]], df["sentiment"], test_size=0.2, random_state=42
    )
    return build_pipeline().fit(X_train, y_train)


model = load_model()

st.markdown('<div class="app-title">IMDB Sentiment Analysis</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Film rəyini yazın, model onun müsbət və ya mənfi olduğunu təyin etsin.</div>',
    unsafe_allow_html=True,
)

review = st.text_area(
    "Rəy",
    height=170,
    placeholder="Məsələn: This movie was absolutely wonderful, the acting was great.",
    label_visibility="collapsed",
)

if st.button("Analiz et", type="primary"):
    if not review.strip():
        st.warning("Əvvəlcə rəy yazın.")
    else:
        X = pd.DataFrame({"review": [review]})
        pred = model.predict(X)[0]
        positive = str(pred).lower() == "positive"

        conf_html = ""
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            conf = proba[list(model.classes_).index(pred)]
            conf_html = f'<div class="result-conf">Əminlik: {conf:.0%}</div>'

        st.markdown(
            f"""
            <div class="result {'positive' if positive else 'negative'}">
              <div class="result-label">Nəticə</div>
              <div class="result-value">{'Müsbət' if positive else 'Mənfi'}</div>
              {conf_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

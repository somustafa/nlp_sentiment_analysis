"""
IMDB Sentiment — Azure ML compute instance üzərində işləyən Streamlit tətbiqi.

Model Azure ML Model Registry-dən yüklənir (imdb-sentiment).
Reyestrə qoşulmaq alınmasa, model buradaca IMDB dataset-dən öyrədilir (~30 san).

İşə salmaq (Azure ML terminalında, layihə qovluğunda):
    pip install streamlit
    streamlit run app_cloud.py --server.port 8501 --server.headless true \
        --server.enableCORS false --server.enableXsrfProtection false
"""
import time

import pandas as pd
import streamlit as st

st.set_page_config(page_title="IMDB Sentiment (Azure)", page_icon="🎬")


@st.cache_resource(show_spinner="Model yüklənir...")
def load_model():
    # 1) Azure ML Model Registry-dən
    try:
        import mlflow

        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions("name='imdb-sentiment'")
        latest = max(int(v.version) for v in versions)
        model = mlflow.sklearn.load_model(f"models:/imdb-sentiment/{latest}")
        return model, f"Azure ML Model Registry (imdb-sentiment v{latest})", None
    except Exception as e:  # reyestr alınmasa, ehtiyat yol
        registry_error = str(e)[:200]

    # 2) Ehtiyat: dataset-dən buradaca öyrət
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
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
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, f"Bu compute-da öyrədildi (accuracy {acc:.3f})", registry_error


model, source, err = load_model()

st.title("IMDB Sentiment Analysis 🎬")
st.caption("Model və tətbiq Azure Machine Learning buludunda işləyir (compute: bulud-cpu)")
st.info(f"Modelin mənbəyi: {source}")

review = st.text_area("Film rəyini yazın (ingiliscə):", height=150)

if st.button("Analiz et", type="primary"):
    if not review.strip():
        st.warning("Əvvəlcə rəy yazın.")
    else:
        start = time.time()
        pred = model.predict(pd.DataFrame({"review": [review]}))[0]
        ms = (time.time() - start) * 1000
        if str(pred).lower() == "positive":
            st.success("Nəticə: Müsbət (positive) 😊")
        else:
            st.error("Nəticə: Mənfi (negative) 😞")
        st.caption(f"Proqnoz müddəti: {ms:.0f} ms")

st.divider()
st.markdown(
    "**Necə işləyir:** rəy → Azure compute instance → "
    "TF-IDF + Logistic Regression (Azure Model Registry) → nəticə"
)

# IMDB Sentiment Analysis

Predicts whether a movie review is **positive** or **negative**, using **TF-IDF + Logistic Regression** trained on 50,000 IMDB reviews, with a **Streamlit** web interface.

**Live demo:** https://YOUR-APP-NAME.streamlit.app

The same app runs in three environments:

| Environment | Where the code runs | How the model is loaded |
|---|---|---|
| Local computer | Your own machine | `model.pkl` or trained on first start |
| Microsoft Azure ML | Azure compute instance (VM in the cloud) | Azure ML **Model Registry** |
| Streamlit Community Cloud | Streamlit's servers (permanent public link) | Trained on first start, then cached |

## Results

| Metric | Value |
|---|---|
| Accuracy | 0.894 |
| F1 Score | 0.897 |

Test set: 10,000 reviews (20% hold-out split).

## Run locally

```bash
git clone https://github.com/somustafa/sentiment_analysis_streamlit_app.git
cd sentiment_analysis_streamlit_app
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501. On the first start it trains the model (takes a few seconds). Optionally, run `python train_test.py` first to save `model.pkl`.

## Run on Microsoft Azure Machine Learning

1. Create an Azure ML workspace and a CPU compute instance.
2. In the compute instance terminal, clone this repo and train the model in the cloud. This logs metrics and registers the model as `imdb-sentiment` in the Azure ML Model Registry:
   ```bash
   pip install "mlflow<3" azureml-mlflow
   python train_azure.py
   ```
3. Start the app on the compute instance:
   ```bash
   pip install streamlit
   streamlit run app.py --server.port 8501 --server.headless true --server.enableCORS false --server.enableXsrfProtection false
   ```
4. Open `https://<compute-name>-8501.<region>.instances.azureml.ms`.

## Project structure

```
app.py              Streamlit app (local / Azure / Streamlit Cloud)
train_test.py       Local training, saves model.pkl
train_azure.py      Cloud training on Azure ML, logs metrics, registers the model
requirements.txt    Dependencies
IMDB Dataset.csv    50,000 labelled IMDB reviews
```

## Tech stack

Python · scikit-learn · pandas · Streamlit · MLflow · Microsoft Azure Machine Learning

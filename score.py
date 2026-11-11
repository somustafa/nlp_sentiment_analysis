"""
Ehtiyat plan: Azure-un hazır sklearn mühiti ilə deploy üçün scoring script.
Bu yolda Azure "qutu" qurmur (image build yoxdur), hazır qutudan istifadə edir.

Gözlənilən sorğu:
  {"input_data": {"columns": ["review"], "data": [["This movie was amazing"]]}}
və ya sadə:
  {"reviews": ["This movie was amazing", "Boring film"]}
"""
import glob
import json
import os
import pickle

import pandas as pd

model = None


def init():
    global model
    model_dir = os.getenv("AZUREML_MODEL_DIR", ".")
    paths = glob.glob(os.path.join(model_dir, "**", "model.pkl"), recursive=True)
    if not paths:
        raise FileNotFoundError(f"model.pkl tapılmadı: {model_dir}")
    with open(paths[0], "rb") as f:
        model = pickle.load(f)


def run(raw_data):
    data = json.loads(raw_data)
    if "input_data" in data:
        texts = [row[0] for row in data["input_data"]["data"]]
    else:
        texts = data["reviews"]
    preds = model.predict(pd.DataFrame({"review": texts}))
    return preds.tolist()

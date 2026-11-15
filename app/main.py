from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI

from app.schemas import HeartFeatures, PredictionResponse

artifact = joblib.load(Path(__file__).resolve().parent.parent / "model" / "heart_model.joblib")
model = artifact["model"]
FEATURES = artifact["features"]

app = FastAPI(title="Heart Disease Prediction API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info")
def info():
    return {"model_type": type(model).__name__, "features": FEATURES}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: HeartFeatures):
    row = pd.DataFrame([payload.model_dump()])[FEATURES]
    return {"heart_disease": bool(model.predict(row)[0] == 1)}

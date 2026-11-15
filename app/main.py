from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.schemas import HeartFeatures, PredictionResponse

BASE_DIR = Path(__file__).resolve().parent.parent
artifact = joblib.load(BASE_DIR / "model" / "heart_model.joblib")
model = artifact["model"]
FEATURES = artifact["features"]

app = FastAPI(title="Heart Disease Prediction API")


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(BASE_DIR / "templates" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info")
def info():
    return {"model_type": type(model).__name__, "features": FEATURES}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: HeartFeatures):
    row = pd.DataFrame([payload.model_dump()])[FEATURES]
    proba = float(model.predict_proba(row)[0][1])
    return {"heart_disease": proba >= 0.5, "probability": round(proba, 4)}

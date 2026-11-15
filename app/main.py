from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import HeartInput, PredictionOut

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "heart_model.joblib"
FEATURES = list(HeartInput.model_fields.keys())

app = FastAPI(title="Heart Disease Prediction API", version="1.0.0")

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model = None


@app.get("/")
def root():
    return {"message": "Heart Disease API. See /docs for Swagger UI."}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/info")
def info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    final_step = model.steps[-1][1] if hasattr(model, "steps") else model
    return {
        "model_type": type(final_step).__name__,
        "n_features": len(FEATURES),
        "features": FEATURES,
        "target": "heart_disease (true = present, false = absent)",
    }


@app.post("/predict", response_model=PredictionOut)
def predict(payload: HeartInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    row = pd.DataFrame([payload.model_dump()])[FEATURES]
    pred = int(model.predict(row)[0])
    proba = float(model.predict_proba(row)[0][1])
    return PredictionOut(heart_disease=bool(pred), probability=round(proba, 4))

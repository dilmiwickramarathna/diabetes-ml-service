# src/api.py
from fastapi import FastAPI, HTTPException
import joblib
import numpy as np

app = FastAPI(title="Virtual Diabetes Clinic ML Service", version="v0.1")

# Load model
try:
    artifacts = joblib.load("artifacts/model.pkl")
    model = artifacts["model"]
    scaler = artifacts["scaler"]
except Exception as e:
    raise RuntimeError(f"Model not found. Train the model first. {e}")

@app.get("/health")
def health():
    return {"status": "ok", "model_version": "v0.1"}

@app.post("/predict")
def predict(features: dict):
    try:
        # Validate input
        X = np.array([[features[f] for f in features]])
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        return {"prediction": float(pred)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

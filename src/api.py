# src/api_v2.py
from fastapi import FastAPI, HTTPException
import joblib
import numpy as np

app = FastAPI(title="Virtual Diabetes Clinic ML Service", version="v0.2")

# Load model
try:
    artifacts = joblib.load("artifacts/model_v2.pkl")
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    selector = artifacts["selector"]
except Exception as e:
    raise RuntimeError(f"Model not found. Train the model first. {e}")


@app.get("/health")
def health():
    return {"status": "ok", "model_version": "v0.2"}


@app.post("/predict")
def predict(features: dict, high_risk_threshold: float = None):
    try:
        X = np.array([[features[f] for f in features]])
        X_scaled = scaler.transform(X)
        X_selected = selector.transform(X_scaled)
        pred = model.predict(X_selected)[0]

        result = {"prediction": float(pred)}
        if high_risk_threshold is not None:
            result["high_risk"] = pred >= high_risk_threshold

        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

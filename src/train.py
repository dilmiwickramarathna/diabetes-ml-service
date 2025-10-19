# src/train.py
import joblib
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import json
import os

def train_model(random_state=42):
    # Load dataset
    Xy = load_diabetes(as_frame=True)
    X = Xy.frame.drop(columns=["target"])
    y = Xy.frame["target"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Evaluate model
    preds = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    # Save artifacts
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump({"scaler": scaler, "model": model}, "artifacts/model.pkl")

    metrics = {"rmse": rmse}
    with open("artifacts/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Model trained. RMSE = {rmse:.2f}")

if __name__ == "__main__":
    train_model()

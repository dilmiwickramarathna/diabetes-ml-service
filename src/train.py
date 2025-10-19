# src/train_v2.py
import joblib
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, precision_score, recall_score
from sklearn.feature_selection import SelectKBest, f_regression
import json
import os


def train_model_v2(model_type="ridge", threshold=None, random_state=42):
    # Load dataset
    Xy = load_diabetes(as_frame=True)
    X = Xy.frame.drop(columns=["target"])
    y = Xy.frame["target"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Optional: feature selection (select top 8 features)
    selector = SelectKBest(score_func=f_regression, k=8)
    X_train_scaled = selector.fit_transform(X_train_scaled, y_train)
    X_test_scaled = selector.transform(X_test_scaled)

    # Choose model
    if model_type == "ridge":
        model = Ridge(alpha=1.0, random_state=random_state)
    elif model_type == "rf":
        model = RandomForestRegressor(n_estimators=100,
                                      random_state=random_state)
    else:
        raise ValueError("model_type must be 'ridge' or 'rf'")

    # Train model
    model.fit(X_train_scaled, y_train)

    # Predict and evaluate
    preds = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    metrics = {"rmse": rmse}

    # Optional: high-risk flag
    if threshold is not None:
        y_true_flag = (y_test >= threshold).astype(int)
        y_pred_flag = (preds >= threshold).astype(int)
        precision = precision_score(y_true_flag, y_pred_flag)
        recall = recall_score(y_true_flag, y_pred_flag)
        metrics.update({"precision": precision,
                        "recall": recall, "threshold": threshold})

    # Save artifacts
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump({"scaler": scaler, "selector": selector,
                 "model": model}, "artifacts/model_v2.pkl")

    with open("artifacts/metrics_v2.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Model trained ({model_type}). RMSE={rmse:.2f}")
    if threshold:
        print(f"High-risk flag -> Precision={precision:.2f}, "
              f"Recall={recall:.2f}")


if __name__ == "__main__":
    # Example: Ridge regression with high-risk threshold = 200
    train_model_v2(model_type="ridge", threshold=200)

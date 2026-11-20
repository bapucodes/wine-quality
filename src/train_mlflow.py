import json
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

try:
    import mlflow
    import mlflow.sklearn
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


def train_and_track(run_name="RandomForest_WineQuality", params=None):
    if params is None:
        params = {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
            "class_weight": "balanced"
        }

    print(f"\n--- Starting MLflow Run: {run_name} ---")

    X_train = np.load("data/processed/X_train_final.npy")
    X_test = np.load("data/processed/X_test_final.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_test = np.load("data/processed/y_test.npy")

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/wine_quality_model.pkl")

    if HAS_MLFLOW:
        try:
            mlflow.set_experiment("Wine_Quality_Prediction")
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(model, "model")
        except Exception as e:
            print(f"[INFO] MLflow note: {e}")

    print("--- Metrics ---")
    for k, v in metrics.items():
        print(f"  {k:10s}: {v:.4f}")

    return metrics


if __name__ == "__main__":
    train_and_track()

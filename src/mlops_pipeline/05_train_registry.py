import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

try:
    import mlflow
    import mlflow.sklearn
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


def configure_mlflow(config):
    if not HAS_MLFLOW:
        return
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config["mlflow"].get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)


def train_and_register_model():
    print("[INFO] --- Starting MLflow Run with Model Registry - Wine Quality MLOps ---")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    processed_dir = config["data"]["processed_dir"]

    try:
        X_train = np.load(os.path.join(processed_dir, "X_train_final.npy"))
        X_test = np.load(os.path.join(processed_dir, "X_test_final.npy"))
        y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
        y_test = np.load(os.path.join(processed_dir, "y_test.npy"))
    except FileNotFoundError:
        print("[ERROR] Processed data not found. Please run 03_preprocessing.py first.")
        return False

    model_cfg = config["model"]

    params = {
        "n_estimators": model_cfg["n_estimators"],
        "max_depth": model_cfg["max_depth"],
        "random_state": model_cfg["random_state"],
        "class_weight": model_cfg["class_weight"],
    }

    print("[INFO] Training RandomForest model...")
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    print("[INFO] Evaluating model on the held-out test set...")
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
    os.makedirs("artifacts", exist_ok=True)

    joblib.dump(model, config["paths"]["model"])
    joblib.dump(model, "models/random_forest_model.pkl")
    joblib.dump(model, "models/random_forest_baseline.pkl")
    joblib.dump(0.5, "models/wine_decision_threshold.pkl")

    run_id = "run_wine_quality_local_01"

    if HAS_MLFLOW:
        try:
            configure_mlflow(config)
            mlflow.set_experiment(config["mlflow"]["experiment_name"])
            with mlflow.start_run(run_name="RandomForest_WineQuality") as run:
                run_id = run.info.run_id
                mlflow.log_params(params)
                mlflow.log_param("model_family", "RandomForest")
                mlflow.log_param("lab", "ML Pipeline")
                mlflow.log_metrics(metrics)

                preprocessor_path = config["paths"]["preprocessor"]
                if os.path.exists(preprocessor_path):
                    mlflow.log_artifact(preprocessor_path, artifact_path="preprocessing_pipeline")

                print("[INFO] Pushing model to MLflow Registry...")
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="random_forest_model",
                    registered_model_name=config["registry"]["model_name"],
                )
        except Exception as exc:
            print(f"[WARNING] MLflow logging skipped: {exc}")
    else:
        print("[INFO] MLflow not installed. Model and metrics saved to local artifacts.")

    # Save production model report in artifacts
    prod_report = {
        "model_name": config["registry"]["model_name"],
        "model_type": "RandomForestClassifier",
        "run_id": run_id,
        "parameters": params,
        "metrics": {k: round(v, 4) for k, v in metrics.items()},
        "model_path": config["paths"]["model"]
    }
    with open("artifacts/production_model_report.json", "w", encoding="utf-8") as f:
        json.dump(prod_report, f, indent=4)

    print(
        f"[SUCCESS] Accuracy={metrics['accuracy']:.4f} | "
        f"Recall={metrics['recall']:.4f} | "
        f"F1={metrics['f1_score']:.4f} | "
        f"ROC-AUC={metrics['roc_auc']:.4f}"
    )
    print(f"[SUCCESS] Model saved to {config['paths']['model']}")
    print(f"[SUCCESS] Model registered as '{config['registry']['model_name']}'")
    print(f"[INFO] Run ID: {run_id}")

    return True


if __name__ == "__main__":
    passed = train_and_register_model()
    sys.exit(0 if passed else 1)

import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

UPDATED_DATASET_PATH = Path("data/processed/updated_training_data.csv")
TEST_FEATURES_PATH = Path("data/processed/X_test_raw.csv")
TEST_LABELS_PATH = Path("data/processed/y_test.npy")

MODEL_V2_PATH = Path("models/retrained_model_v2.pkl")
PREPROCESSOR_V2_PATH = Path("models/retrained_preprocessor_v2.pkl")
REPORT_PATH = Path("outputs/retraining/retraining_report.json")


def clean_dataframe(df, target_column="quality", threshold=6):
    df = df.copy()
    if "wine_id" in df.columns:
        df = df.drop(columns=["wine_id"])
    if target_column in df.columns:
        df[target_column] = df[target_column].apply(
            lambda x: 1 if float(x) >= threshold else 0
        ).astype(int)
    return df


def main():
    print("=" * 70)
    print("RETRAIN MODEL V2 - WINE QUALITY MLOPS")
    print("=" * 70)

    if not UPDATED_DATASET_PATH.exists():
        # Fallback to post_drift or raw data
        train_df = pd.read_csv("data/raw/winequality.csv")
    else:
        train_df = pd.read_csv(UPDATED_DATASET_PATH)

    clean_train = clean_dataframe(train_df, "quality")
    X_train = clean_train.drop(columns=["quality"])
    y_train = clean_train["quality"]

    test_features = pd.read_csv(TEST_FEATURES_PATH)
    y_test = np.load(TEST_LABELS_PATH)
    clean_test = clean_dataframe(test_features, "quality")

    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_cols)
        ]
    )

    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(clean_test)

    model_v2 = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, class_weight="balanced")
    model_v2.fit(X_train_trans, y_train)

    y_pred = model_v2.predict(X_test_trans)
    y_proba = model_v2.predict_proba(X_test_trans)[:, 1]

    metrics_v2 = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    MODEL_V2_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_v2, MODEL_V2_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_V2_PATH)

    report = {
        "stage": "Retraining V2",
        "model_version": "v2",
        "training_samples": len(X_train),
        "metrics": metrics_v2,
        "model_path": str(MODEL_V2_PATH).replace("\\", "/"),
        "preprocessor_path": str(PREPROCESSOR_V2_PATH).replace("\\", "/")
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"[SUCCESS] Retrained Model V2 saved to {MODEL_V2_PATH}")
    print(f"[SUCCESS] Metrics: {metrics_v2}")


if __name__ == "__main__":
    main()

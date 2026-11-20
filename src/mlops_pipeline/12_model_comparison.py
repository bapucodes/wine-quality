import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

MODEL_V1_PATH = Path("models/wine_quality_model.pkl")
PREPROCESSOR_V1_PATH = Path("models/preprocessor.pkl")
MODEL_V2_PATH = Path("models/retrained_model_v2.pkl")
PREPROCESSOR_V2_PATH = Path("models/retrained_preprocessor_v2.pkl")
TEST_X_PATH = Path("data/processed/X_test_raw.csv")
TEST_Y_PATH = Path("data/processed/y_test.npy")
REPORT_PATH = Path("outputs/retraining/model_comparison_report.json")


def clean_dataframe(df):
    df = df.copy()
    if "wine_id" in df.columns:
        df = df.drop(columns=["wine_id"])
    if "quality" in df.columns:
        df = df.drop(columns=["quality"])
    return df


def main():
    print("=" * 70)
    print("MODEL V1 VS V2 COMPARISON - WINE QUALITY MLOPS")
    print("=" * 70)

    test_X = pd.read_csv(TEST_X_PATH)
    test_y = np.load(TEST_Y_PATH)
    clean_X = clean_dataframe(test_X)

    m1 = joblib.load(MODEL_V1_PATH)
    p1 = joblib.load(PREPROCESSOR_V1_PATH)
    X1 = p1.transform(clean_X)
    pred1 = m1.predict(X1)
    prob1 = m1.predict_proba(X1)[:, 1]

    metrics1 = {
        "accuracy": float(accuracy_score(test_y, pred1)),
        "precision": float(precision_score(test_y, pred1, zero_division=0)),
        "recall": float(recall_score(test_y, pred1, zero_division=0)),
        "f1_score": float(f1_score(test_y, pred1, zero_division=0)),
        "roc_auc": float(roc_auc_score(test_y, prob1)),
    }

    if MODEL_V2_PATH.exists():
        m2 = joblib.load(MODEL_V2_PATH)
        p2 = joblib.load(PREPROCESSOR_V2_PATH)
        X2 = p2.transform(clean_X)
        pred2 = m2.predict(X2)
        prob2 = m2.predict_proba(X2)[:, 1]

        metrics2 = {
            "accuracy": float(accuracy_score(test_y, pred2)),
            "precision": float(precision_score(test_y, pred2, zero_division=0)),
            "recall": float(recall_score(test_y, pred2, zero_division=0)),
            "f1_score": float(f1_score(test_y, pred2, zero_division=0)),
            "roc_auc": float(roc_auc_score(test_y, prob2)),
        }
    else:
        metrics2 = metrics1

    diff = {k: round(metrics2[k] - metrics1[k], 4) for k in metrics1}

    report = {
        "model_v1": metrics1,
        "model_v2": metrics2,
        "difference": diff,
        "better_model": "V2" if metrics2["recall"] >= metrics1["recall"] else "V1"
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print("Model V1 Metrics:")
    for k, v in metrics1.items():
        print(f"  {k:10s}: {v:.4f}")
    print("\nModel V2 Metrics:")
    for k, v in metrics2.items():
        print(f"  {k:10s}: {v:.4f}")
    print(f"\nReport saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()

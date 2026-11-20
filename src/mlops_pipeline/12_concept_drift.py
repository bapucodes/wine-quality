import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from api.model_loader import ModelLoader
from api.preprocessing import clean_prediction_data

BASELINE_X_PATH = Path("data/processed/X_test_raw.csv")
BASELINE_Y_PATH = Path("data/processed/y_test.npy")
EVOLVED_PATH = Path("data/monitoring/drift/evolved_labeled_data.csv")
REPORT_PATH = Path("outputs/drift/concept_drift_report.csv")


def evaluate(model_loader, X, y):
    X_clean = clean_prediction_data(X)
    X_transformed = model_loader.preprocessor.transform(X_clean)

    predictions = model_loader.model.predict(X_transformed)
    probabilities = model_loader.model.predict_proba(X_transformed)[:, 1]

    return {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y, probabilities),
    }


def main():
    print("=" * 70)
    print("CONCEPT DRIFT ANALYSIS - WINE QUALITY MLOPS")
    print("=" * 70)

    baseline_X = pd.read_csv(BASELINE_X_PATH)
    baseline_y = np.load(BASELINE_Y_PATH)

    evolved = pd.read_csv(EVOLVED_PATH)

    # In winequality, target is quality (binarized >= 6)
    target_col = "quality"
    evolved_X = evolved.drop(columns=[target_col])
    evolved_y = evolved[target_col].apply(lambda x: 1 if float(x) >= 6 else 0).to_numpy()

    model_loader = ModelLoader()
    model_loader.load()

    baseline_metrics = evaluate(model_loader, baseline_X, baseline_y)
    evolved_metrics = evaluate(model_loader, evolved_X, evolved_y)

    print()
    print("BASELINE PERFORMANCE:")
    for metric, value in baseline_metrics.items():
        print(f"  {metric.upper():10s}: {value:.4f}")

    print()
    print("EVOLVED PERFORMANCE:")
    for metric, value in evolved_metrics.items():
        print(f"  {metric.upper():10s}: {value:.4f}")

    results = []
    for metric in baseline_metrics:
        b_val = baseline_metrics[metric]
        e_val = evolved_metrics[metric]
        change = e_val - b_val
        results.append({
            "metric": metric,
            "baseline": round(b_val, 4),
            "evolved": round(e_val, 4),
            "change": round(change, 4),
        })

    report = pd.DataFrame(results)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT_PATH, index=False)

    print()
    print("=" * 70)
    print("PERFORMANCE CHANGE")
    print("=" * 70)
    print(report.to_string(index=False))
    print(f"\nReport saved : {REPORT_PATH}")


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ks_2samp

BASELINE_PATH = Path("data/processed/X_test_raw.csv")
EVOLVED_PATH = Path("data/monitoring/drift/evolved_data.csv")
REPORT_PATH = Path("outputs/drift/data_drift_report.csv")

NUMERICAL_COLUMNS = [
    "alcohol",
    "volatile acidity",
    "sulphates",
    "citric acid",
    "residual sugar",
    "fixed acidity",
    "pH",
    "density",
    "chlorides",
    "total sulfur dioxide"
]

KS_THRESHOLD = 0.10


def load_data():
    baseline = pd.read_csv(BASELINE_PATH)
    evolved = pd.read_csv(EVOLVED_PATH)

    # Align column subset if needed
    cols = [c for c in NUMERICAL_COLUMNS if c in baseline.columns and c in evolved.columns]
    return baseline[cols], evolved[cols]


def analyze_numerical_drift(baseline, evolved):
    results = []
    for column in baseline.columns:
        statistic, p_value = ks_2samp(
            baseline[column].dropna(),
            evolved[column].dropna(),
        )
        status = "DRIFT" if statistic >= KS_THRESHOLD else "OK"
        results.append({
            "feature": column,
            "type": "numerical",
            "metric": "KS",
            "score": round(float(statistic), 6),
            "p_value": round(float(p_value), 6),
            "threshold": KS_THRESHOLD,
            "status": status,
        })
    return results


def create_drift_report(baseline, evolved):
    results = analyze_numerical_drift(baseline, evolved)
    report = pd.DataFrame(results)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT_PATH, index=False)
    return report


def print_report(report):
    print("=" * 70)
    print("DATA DRIFT ANALYSIS - RED WINE QUALITY")
    print("=" * 70)
    print(f"Baseline dataset : {BASELINE_PATH}")
    print(f"Evolved dataset  : {EVOLVED_PATH}")
    print()
    print(report.to_string(index=False))

    drift_count = (report["status"] == "DRIFT").sum()
    total_features = len(report)

    print()
    print("=" * 70)
    print(f"Features analyzed : {total_features}")
    print(f"Drift detected    : {drift_count}")
    print(f"Report saved      : {REPORT_PATH}")
    print("=" * 70)


def main():
    baseline, evolved = load_data()
    print(f"Baseline shape : {baseline.shape}")
    print(f"Evolved shape  : {evolved.shape}")
    print()
    report = create_drift_report(baseline, evolved)
    print_report(report)


if __name__ == "__main__":
    main()

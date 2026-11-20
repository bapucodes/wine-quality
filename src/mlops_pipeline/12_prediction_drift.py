import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from api.model_loader import ModelLoader
from api.preprocessing import clean_prediction_data

BASELINE_PATH = Path("data/processed/X_test_raw.csv")
EVOLVED_PATH = Path("data/monitoring/drift/evolved_data.csv")
REPORT_PATH = Path("outputs/drift/prediction_drift_report.csv")

KS_THRESHOLD = 0.10


def load_data():
    baseline = pd.read_csv(BASELINE_PATH)
    evolved = pd.read_csv(EVOLVED_PATH)
    return baseline, evolved


def predict(model_loader, data):
    clean_data = clean_prediction_data(data)
    transformed_data = model_loader.preprocessor.transform(clean_data)

    predictions = model_loader.model.predict(transformed_data)
    probabilities = model_loader.model.predict_proba(transformed_data)[:, 1]

    return predictions, probabilities


def summarize_predictions(name, predictions, probabilities):
    print()
    print(f"{name} PREDICTIONS")
    print("-" * 60)
    total = len(predictions)
    good_count = int((predictions == 1).sum())
    normal_count = int((predictions == 0).sum())

    print(f"Total predictions     : {total}")
    print(f"Good Quality (1)      : {good_count} ({good_count / total * 100:.2f}%)")
    print(f"Normal/Standard (0)   : {normal_count} ({normal_count / total * 100:.2f}%)")
    print(f"Mean probability      : {probabilities.mean():.6f}")
    print(f"Median probability    : {np.median(probabilities):.6f}")
    print(f"Std probability       : {probabilities.std():.6f}")


def analyze_prediction_drift(baseline_predictions, evolved_predictions, baseline_probabilities, evolved_probabilities):
    prediction_ks, prediction_p = ks_2samp(baseline_predictions, evolved_predictions)
    probability_ks, probability_p = ks_2samp(baseline_probabilities, evolved_probabilities)

    prediction_drift = prediction_ks >= KS_THRESHOLD
    probability_drift = probability_ks >= KS_THRESHOLD

    return {
        "prediction_distribution_ks": float(prediction_ks),
        "prediction_distribution_p_value": float(prediction_p),
        "prediction_distribution_status": "DRIFT" if prediction_drift else "OK",
        "probability_distribution_ks": float(probability_ks),
        "probability_distribution_p_value": float(probability_p),
        "probability_distribution_status": "DRIFT" if probability_drift else "OK",
    }


def save_report(results):
    report = pd.DataFrame([results])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT_PATH, index=False)
    return report


def main():
    print("=" * 70)
    print("PREDICTION DRIFT ANALYSIS - WINE QUALITY MLOPS")
    print("=" * 70)

    baseline, evolved = load_data()
    print(f"Baseline dataset : {BASELINE_PATH} ({baseline.shape})")
    print(f"Evolved dataset  : {EVOLVED_PATH} ({evolved.shape})")

    model_loader = ModelLoader()
    model_loader.load()

    baseline_predictions, baseline_probabilities = predict(model_loader, baseline)
    evolved_predictions, evolved_probabilities = predict(model_loader, evolved)

    summarize_predictions("BASELINE", baseline_predictions, baseline_probabilities)
    summarize_predictions("EVOLVED", evolved_predictions, evolved_probabilities)

    results = analyze_prediction_drift(
        baseline_predictions,
        evolved_predictions,
        baseline_probabilities,
        evolved_probabilities,
    )

    print()
    print("=" * 70)
    print("PREDICTION DRIFT RESULTS")
    print("=" * 70)
    print(f"Prediction distribution KS : {results['prediction_distribution_ks']:.6f} ({results['prediction_distribution_status']})")
    print(f"Probability distribution KS: {results['probability_distribution_ks']:.6f} ({results['probability_distribution_status']})")
    print(f"Threshold                  : KS >= {KS_THRESHOLD}")
    print(f"Report saved               : {REPORT_PATH}")

    save_report(results)


if __name__ == "__main__":
    main()

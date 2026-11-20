import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


def run_evaluation():
    print("Starting Model Evaluation - Red Wine Quality...")

    # 1. Load test data and the trained model
    X_test_final = np.load("data/processed/X_test_final.npy")
    y_test = np.load("data/processed/y_test.npy")
    model = joblib.load("models/random_forest_baseline.pkl")

    # 2. Make predictions
    y_pred = model.predict(X_test_final)

    # 3. Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n--- Model Evaluation Report ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("-------------------------------\n")

    # 4. Error Analysis Export
    raw_df = pd.read_csv("data/raw/winequality.csv")
    if "wine_id" in raw_df.columns:
        raw_df = raw_df.drop(columns=["wine_id"])
    raw_df["quality"] = raw_df["quality"].apply(lambda x: 1 if float(x) >= 6 else 0).astype(int)

    _, X_test_raw = train_test_split(
        raw_df.drop("quality", axis=1),
        test_size=0.2,
        random_state=42,
        stratify=raw_df["quality"]
    )

    errors_df = X_test_raw.copy()
    errors_df["Actual_Quality"] = y_test
    errors_df["Predicted_Quality"] = y_pred

    false_negatives = errors_df[(errors_df["Actual_Quality"] == 1) & (errors_df["Predicted_Quality"] == 0)]
    false_positives = errors_df[(errors_df["Actual_Quality"] == 0) & (errors_df["Predicted_Quality"] == 1)]

    import os
    os.makedirs("outputs", exist_ok=True)
    false_negatives.to_csv("outputs/false_negatives.csv", index=False)
    false_positives.to_csv("outputs/false_positives.csv", index=False)

    print("Evaluation complete and error analysis files saved!")


if __name__ == "__main__":
    run_evaluation()

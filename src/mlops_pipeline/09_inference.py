import os
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import yaml

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from validation import NUMERIC_COLUMNS


def clean_dataframe(df, config):
    df = df.copy()
    target_column = config["data"]["target_column"]
    id_column = config["data"].get("id_column", "wine_id")

    if id_column in df.columns:
        df = df.drop(columns=[id_column])
    if target_column in df.columns:
        df = df.drop(columns=[target_column])

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def run_inference_demo():
    print("[INFO] Starting ML Pipeline Production Inference - Wine Quality MLOps...")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Prefer deployment bundle if available, else local models dir
    model_path = "deployment/model.pkl"
    if not os.path.exists(model_path):
        model_path = config["paths"]["model"]

    preprocessor_path = "deployment/preprocessor.pkl"
    if not os.path.exists(preprocessor_path):
        preprocessor_path = config["paths"]["preprocessor"]

    raw_data_path = config["data"]["raw_path"]

    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at {model_path}")
        return False
    if not os.path.exists(preprocessor_path):
        print(f"[ERROR] Preprocessor not found at {preprocessor_path}")
        return False
    if not os.path.exists(raw_data_path):
        print(f"[ERROR] Raw dataset not found at {raw_data_path}")
        return False

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    df = pd.read_csv(raw_data_path)

    sample = df.sample(n=5, random_state=42)
    X_sample = clean_dataframe(sample, config)
    X_transformed = preprocessor.transform(X_sample)

    predictions = model.predict(X_transformed)
    probabilities = model.predict_proba(X_transformed)[:, 1]

    result = sample.copy()
    result["Predicted_Good_Quality"] = predictions
    result["Quality_Probability"] = np.round(probabilities, 4)

    print("\n--- ML Pipeline Production Inference Results ---")
    cols_to_show = ["wine_id", "alcohol", "volatile acidity", "sulphates", "quality", "Predicted_Good_Quality", "Quality_Probability"]
    show_cols = [c for c in cols_to_show if c in result.columns]
    print(result[show_cols].to_string(index=False))
    print("------------------------------------------------\n")

    os.makedirs("outputs", exist_ok=True)
    output_path = config["paths"]["inference_output"]
    result.to_csv(output_path, index=False)

    print(f"[SUCCESS] Predictions saved to {output_path}")
    return True


if __name__ == "__main__":
    passed = run_inference_demo()
    sys.exit(0 if passed else 1)

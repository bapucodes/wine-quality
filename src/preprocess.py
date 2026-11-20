import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def run_preprocessing():
    print("Starting Preprocessing Pipeline - Red Wine Quality...")

    # 1. Load the raw data
    data_path = "data/raw/winequality.csv"
    df = pd.read_csv(data_path)

    if "wine_id" in df.columns:
        df = df.drop(columns=["wine_id"])

    # Binarize quality (>= 6 is positive class 1)
    df["quality"] = df["quality"].apply(lambda x: 1 if float(x) >= 6 else 0).astype(int)

    X = df.drop(columns=["quality"])
    y = df["quality"]

    # 2. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns

    # 3. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[num_cols])
    X_test_scaled = scaler.transform(X_test[num_cols])

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # 4. Save Artifacts
    np.save("data/processed/X_train_final.npy", X_train_scaled)
    np.save("data/processed/X_test_final.npy", X_test_scaled)
    np.save("data/processed/y_train.npy", y_train.to_numpy(dtype=np.int64))
    np.save("data/processed/y_test.npy", y_test.to_numpy(dtype=np.int64))

    X_train.to_csv("data/processed/X_train_raw.csv", index=False)
    X_test.to_csv("data/processed/X_test_raw.csv", index=False)

    joblib.dump(scaler, "models/scaler.pkl")

    # Save Metadata
    metadata = {
        "dataset_name": "Red Wine Quality",
        "train_shape": list(X_train_scaled.shape),
        "test_shape": list(X_test_scaled.shape),
        "numerical_features": list(num_cols),
        "categorical_features": []
    }
    with open("data/processed/dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("Preprocessing completed successfully!")


if __name__ == "__main__":
    run_preprocessing()

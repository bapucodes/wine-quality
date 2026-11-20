import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def run_preprocessing():
    print("[INFO] Starting Sklearn Pipeline Preprocessing - Red Wine Quality...")

    # 1. Load the raw data
    data_path = "data/raw/winequality.csv"
    df = pd.read_csv(data_path)

    if "wine_id" in df.columns:
        df = df.drop(columns=["wine_id"])

    # Binarize quality
    df["quality"] = df["quality"].apply(lambda x: 1 if float(x) >= 6 else 0).astype(int)

    X = df.drop(columns=["quality"])
    y = df["quality"]

    # 2. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # 3. Build Modular Scikit-Learn Pipeline
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
        ]
    )

    # 4. Fit & Transform
    print("[INFO] Fitting and transforming training data...")
    X_train_final = preprocessor.fit_transform(X_train)
    X_test_final = preprocessor.transform(X_test)

    # 5. Save Artifacts
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    np.save("data/processed/X_train_final.npy", X_train_final)
    np.save("data/processed/X_test_final.npy", X_test_final)
    np.save("data/processed/y_train.npy", y_train.to_numpy(dtype=np.int64))
    np.save("data/processed/y_test.npy", y_test.to_numpy(dtype=np.int64))

    joblib.dump(preprocessor, "models/preprocessor.pkl")

    # Save full pipeline with classifier
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight="balanced")
    rf.fit(X_train_final, y_train)
    full_pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", rf)])
    joblib.dump(full_pipe, "models/wine_full_pipeline.pkl")

    # Save Metadata
    metadata = {
        "dataset_name": "Red Wine Quality",
        "train_shape": list(X_train_final.shape),
        "test_shape": list(X_test_final.shape),
        "numerical_features": num_cols,
        "categorical_features": []
    }
    with open("data/processed/dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("[SUCCESS] Preprocessing completed successfully! Pipeline saved as preprocessor.pkl.")


if __name__ == "__main__":
    run_preprocessing()

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def clean_dataframe(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Apply deterministic dataset cleaning shared by training and inference."""
    df = df.copy()
    target_column = config["data"]["target_column"]
    id_column = config["data"].get("id_column", "wine_id")
    threshold = config["data"].get("quality_threshold", 6)

    if id_column in df.columns:
        df = df.drop(columns=[id_column])

    # Convert numeric features
    numeric_cols = [
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
        "chlorides", "free sulfur dioxide", "total sulfur dioxide",
        "density", "pH", "sulphates", "alcohol"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Map quality to binary integer
    if target_column in df.columns:
        def binarize(val):
            try:
                num = float(val)
                return 1 if num >= threshold else 0
            except (ValueError, TypeError):
                s = str(val).strip().lower()
                return 1 if s in ["1", "yes", "true", "good"] else 0

        df[target_column] = df[target_column].apply(binarize).astype(int)

    return df


def create_initial_test_ids(df: pd.DataFrame, config: dict, processed_dir: str):
    """Recreate the original test split and permanently save its wine IDs."""
    id_column = config["data"].get("id_column", "wine_id")
    target_column = config["data"]["target_column"]
    threshold = config["data"].get("quality_threshold", 6)
    test_ids_path = os.path.join(processed_dir, "test_customer_ids.csv")
    test_wine_ids_path = os.path.join(processed_dir, "test_wine_ids.csv")

    for path in [test_ids_path, test_wine_ids_path]:
        if os.path.exists(path):
            test_ids = pd.read_csv(path)
            col = id_column if id_column in test_ids.columns else test_ids.columns[0]
            return test_ids[col].astype(str).tolist()

    print("[INFO] test_wine_ids.csv not found.")
    print("[INFO] Recreating the original train/test split to recover fixed test IDs...")

    working_df = df.copy()
    working_df[id_column] = working_df[id_column].astype(str)

    def binarize(val):
        try:
            return 1 if float(val) >= threshold else 0
        except (ValueError, TypeError):
            return 1 if str(val).strip().lower() in ["1", "yes", "true", "good"] else 0

    working_df[target_column] = working_df[target_column].apply(binarize).astype(int)

    X = working_df.drop(columns=[id_column, target_column])
    y = working_df[target_column]

    _, X_test, _, _ = train_test_split(
        X,
        y,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
        stratify=y,
    )

    test_ids = working_df.loc[X_test.index, id_column].astype(str).tolist()

    pd.DataFrame({id_column: test_ids}).to_csv(test_wine_ids_path, index=False)
    # Also save as test_customer_ids.csv for compatibility with student handout naming
    pd.DataFrame({id_column: test_ids}).to_csv(test_ids_path, index=False)

    print(f"[SUCCESS] Original test-set IDs saved to {test_wine_ids_path}")
    print(f"[INFO] Fixed test set size: {len(test_ids)}")

    return test_ids


def split_raw_data(df: pd.DataFrame, config: dict, processed_dir: str):
    """Keep the original test set fixed while using all other records for training."""
    target_column = config["data"]["target_column"]
    id_column = config["data"].get("id_column", "wine_id")
    threshold = config["data"].get("quality_threshold", 6)

    test_ids_path = os.path.join(processed_dir, "test_wine_ids.csv")
    if not os.path.exists(test_ids_path):
        test_ids_path = os.path.join(processed_dir, "test_customer_ids.csv")

    if os.path.exists(test_ids_path):
        test_ids_df = pd.read_csv(test_ids_path)
        col = id_column if id_column in test_ids_df.columns else test_ids_df.columns[0]
        test_ids = set(test_ids_df[col].astype(str))
        print(f"[INFO] Using fixed test set from {test_ids_path}")
    else:
        test_ids = set(create_initial_test_ids(df, config, processed_dir))
        print("[INFO] Original test set has been permanently fixed.")

    working_df = df.copy()
    working_df[id_column] = working_df[id_column].astype(str)

    test_mask = working_df[id_column].isin(test_ids)
    if not test_mask.any():
        raise ValueError("None of the fixed test IDs were found in the current dataset.")

    test_df = working_df.loc[test_mask].copy()
    train_df = working_df.loc[~test_mask].copy()

    def binarize(val):
        try:
            return 1 if float(val) >= threshold else 0
        except (ValueError, TypeError):
            return 1 if str(val).strip().lower() in ["1", "yes", "true", "good"] else 0

    y_test = test_df[target_column].apply(binarize).astype(int)
    y_train = train_df[target_column].apply(binarize).astype(int)

    X_test = clean_dataframe(test_df, config).drop(columns=[target_column])
    X_train = clean_dataframe(train_df, config).drop(columns=[target_column])

    print(f"[INFO] Training records: {len(X_train)}")
    print(f"[INFO] Fixed test records: {len(X_test)}")

    return X_train, X_test, y_train, y_test


def build_preprocessor(X_train: pd.DataFrame):
    """Create the canonical feature-preprocessing pipeline."""
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    transformers = [("num", num_pipeline, num_cols)]
    if cat_cols:
        from sklearn.preprocessing import OneHotEncoder
        cat_pipeline = Pipeline(
            steps=[
                ("ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat", cat_pipeline, cat_cols))

    return ColumnTransformer(transformers=transformers), cat_cols, num_cols


def run_preprocessing():
    print("[INFO] Starting Preprocessing - Wine Quality MLOps...")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_path = config["data"]["raw_path"]
    processed_dir = config["data"]["processed_dir"]
    target_column = config["data"]["target_column"]

    if not os.path.exists(data_path):
        print(f"[ERROR] Raw dataset not found at: {data_path}")
        return False

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    raw_df = pd.read_csv(data_path)
    X_train, X_test, y_train, y_test = split_raw_data(raw_df, config, processed_dir)

    preprocessor, cat_cols, num_cols = build_preprocessor(X_train)

    print("[INFO] Fitting preprocessing pipeline on training data...")
    X_train_final = preprocessor.fit_transform(X_train)
    X_test_final = preprocessor.transform(X_test)

    np.save(os.path.join(processed_dir, "X_train_final.npy"), X_train_final)
    np.save(os.path.join(processed_dir, "X_test_final.npy"), X_test_final)
    np.save(os.path.join(processed_dir, "y_train.npy"), y_train.to_numpy(dtype=np.int64))
    np.save(os.path.join(processed_dir, "y_test.npy"), y_test.to_numpy(dtype=np.int64))

    X_train.to_csv(os.path.join(processed_dir, "X_train_raw.csv"), index=False)
    X_test.to_csv(os.path.join(processed_dir, "X_test_raw.csv"), index=False)

    # Updated training data (X + y combined)
    updated_train = X_train.copy()
    updated_train[target_column] = y_train.values
    updated_train.to_csv(os.path.join(processed_dir, "updated_training_data.csv"), index=False)

    joblib.dump(preprocessor, config["paths"]["preprocessor"])

    # Extract scaler if present
    try:
        scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]
        joblib.dump(scaler, "models/scaler.pkl")
    except Exception:
        pass

    metadata = {
        "dataset_name": "Red Wine Quality",
        "target_column": target_column,
        "train_shape_raw": list(X_train.shape),
        "test_shape_raw": list(X_test.shape),
        "train_shape_transformed": list(X_train_final.shape),
        "test_shape_transformed": list(X_test_final.shape),
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
        "split_test_size": config["split"]["test_size"],
        "split_random_state": config["split"]["random_state"],
        "fixed_test_set": True,
        "test_wine_ids_path": os.path.join(processed_dir, "test_wine_ids.csv"),
        "raw_train_path": os.path.join(processed_dir, "X_train_raw.csv"),
        "raw_test_path": os.path.join(processed_dir, "X_test_raw.csv"),
    }

    with open(os.path.join(processed_dir, "dataset_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    summary_report = {
        "stage": "Preprocessing",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "features_count": len(num_cols) + len(cat_cols),
        "preprocessor_saved": config["paths"]["preprocessor"],
        "status": "PASSED"
    }
    with open("artifacts/preprocessing_summary_report.json", "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=4)

    print(f"[SUCCESS] Preprocessing complete. Preprocessor saved to {config['paths']['preprocessor']}")
    return True


if __name__ == "__main__":
    passed = run_preprocessing()
    sys.exit(0 if passed else 1)

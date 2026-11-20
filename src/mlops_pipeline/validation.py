import os
import pandas as pd

EXPECTED_COLUMNS = [
    "wine_id",
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "quality",
]

NUMERIC_COLUMNS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]


def validate_schema(df: pd.DataFrame, output_report_name: str = "schema_validation_errors.csv") -> bool:
    """Validate wine quality dataset schema and value constraints."""
    print(f"[INFO] Validating schema (Records: {len(df)})...")
    errors = []

    # 1. Column existence
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            errors.append({
                "schema_context": "DataFrameSchema",
                "column": col,
                "check": "column_exists",
                "failure_case": "missing_column",
                "index": None,
            })

    if errors:
        return _handle_validation_errors(errors, output_report_name)

    # 2. Check wine_id
    if df["wine_id"].isnull().any():
        null_indices = df[df["wine_id"].isnull()].index.tolist()
        for idx in null_indices[:10]:
            errors.append({
                "schema_context": "Column",
                "column": "wine_id",
                "check": "not_nullable",
                "failure_case": "NaN",
                "index": idx,
            })

    # 3. Numeric constraints
    for col in NUMERIC_COLUMNS:
        # Check non-numeric
        coerced = pd.to_numeric(df[col], errors="coerce")
        bad_mask = coerced.isna() & df[col].notna()
        if bad_mask.any():
            for idx, val in df[bad_mask][[col]].head(10).itertuples():
                errors.append({
                    "schema_context": "Column",
                    "column": col,
                    "check": "dtype('float64')",
                    "failure_case": str(val),
                    "index": idx,
                })
        # Check negative
        neg_mask = coerced < 0.0
        if neg_mask.any():
            for idx, val in df[neg_mask][[col]].head(10).itertuples():
                errors.append({
                    "schema_context": "Column",
                    "column": col,
                    "check": "greater_than_or_equal_to(0.0)",
                    "failure_case": str(val),
                    "index": idx,
                })

    # 4. Check pH bounds (e.g. 1.0 to 14.0)
    coerced_ph = pd.to_numeric(df["pH"], errors="coerce")
    bad_ph = (coerced_ph < 1.0) | (coerced_ph > 14.0)
    if bad_ph.any():
        for idx, val in df[bad_ph][["pH"]].head(10).itertuples():
            errors.append({
                "schema_context": "Column",
                "column": "pH",
                "check": "in_range(1.0, 14.0)",
                "failure_case": str(val),
                "index": idx,
            })

    # 5. Check quality
    coerced_q = pd.to_numeric(df["quality"], errors="coerce")
    bad_q = coerced_q.isna()
    if bad_q.any():
        for idx, val in df[bad_q][["quality"]].head(10).itertuples():
            errors.append({
                "schema_context": "Column",
                "column": "quality",
                "check": "valid_quality_score",
                "failure_case": str(val),
                "index": idx,
            })

    if errors:
        return _handle_validation_errors(errors, output_report_name)

    print("[SUCCESS] Schema Validation PASSED. Dataset is clean.")
    return True


def _handle_validation_errors(errors: list, output_report_name: str) -> bool:
    print("[ERROR] Schema Validation FAILED. Corruptions detected.")
    failures_df = pd.DataFrame(errors)
    print(failures_df.head(10).to_string())

    os.makedirs("artifacts", exist_ok=True)
    report_path = os.path.join("artifacts", output_report_name)
    failures_df.to_csv(report_path, index=False)
    print(f"[INFO] Detailed failure report saved to '{report_path}'.")
    return False

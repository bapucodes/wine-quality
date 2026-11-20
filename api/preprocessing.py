import pandas as pd


FEATURES = [
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
    "alcohol"
]

RENAME_MAP = {
    "fixed_acidity": "fixed acidity",
    "volatile_acidity": "volatile acidity",
    "citric_acid": "citric acid",
    "residual_sugar": "residual sugar",
    "free_sulfur_dioxide": "free sulfur dioxide",
    "total_sulfur_dioxide": "total sulfur dioxide",
}


def clean_prediction_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw wine data before prediction.
    Ensures correct column names, types, and removes non-feature columns.
    """
    df = df.copy()

    # Rename snake_case to space-separated column names if necessary
    df = df.rename(columns=RENAME_MAP)

    # Remove identifier or target columns
    for col in ["wine_id", "customerID", "quality"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Convert all feature columns to numeric
    for col in df.columns:
        if col in FEATURES:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Reorder columns to expected feature order if all features present
    existing_features = [col for col in FEATURES if col in df.columns]
    if len(existing_features) == len(FEATURES):
        df = df[FEATURES]

    return df

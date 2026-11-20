import pandas as pd
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from validate_data import validate_schema


def simulate_corrupted_data(df):
    """Simulates real-world data corruption scenarios across multiple columns."""
    print("[INFO] Injecting simulated faults into wine quality dataset...")
    corrupted_df = df.astype(object).copy()

    # 1. Invalid numerical values (negative alcohol)
    corrupted_df.loc[0, "alcohol"] = -5.0

    # 2. Invalid pH range
    corrupted_df.loc[1, "pH"] = 25.0

    # 3. Invalid target label
    corrupted_df.loc[2, "quality"] = "Superb"

    # 4. Datatype mismatch (injecting string into float column)
    corrupted_df.loc[3, "fixed acidity"] = "Heavy Acid"

    # 5. Schema inconsistency (unexpected extra column)
    corrupted_df["unexpected_column"] = 123

    # 6. Missing required attribute (dropping a column)
    corrupted_df = corrupted_df.drop(columns=["sulphates"])

    return corrupted_df


if __name__ == "__main__":
    data_path = "data/raw/winequality.csv"
    try:
        raw_df = pd.read_csv(data_path)
        corrupted_df = simulate_corrupted_data(raw_df)
        print("\n--- Running Validation on Corrupted Data ---")
        validate_schema(corrupted_df, output_report_name="corrupted_simulation_errors.csv")
    except FileNotFoundError:
        print(f"[ERROR] Could not find data file at {data_path}.")

import pandas as pd
from pathlib import Path

BASE_TRAIN_PATH = Path("data/processed/X_train_raw.csv")
LABELS_PATH = Path("data/processed/y_train.npy")
EVOLVED_PATH = Path("data/monitoring/drift/evolved_labeled_data.csv")
OUTPUT_PATH = Path("data/processed/updated_training_data.csv")
POST_DRIFT_PATH = Path("data/monitoring/drift/post_drift_training_data.csv")


def main():
    print("[INFO] Combining baseline training data with incoming post-drift batches...")
    base_X = pd.read_csv(BASE_TRAIN_PATH)
    import numpy as np
    base_y = np.load(LABELS_PATH)
    base_df = base_X.copy()
    base_df["quality"] = base_y

    if EVOLVED_PATH.exists():
        evolved_df = pd.read_csv(EVOLVED_PATH)
        # Take a subset of evolved batch to add to training
        evolved_sample = evolved_df.sample(n=min(200, len(evolved_df)), random_state=42)
        if "wine_id" in evolved_sample.columns and "wine_id" not in base_df.columns:
            evolved_sample = evolved_sample.drop(columns=["wine_id"])
        updated = pd.concat([base_df, evolved_sample], ignore_index=True)
    else:
        updated = base_df

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(OUTPUT_PATH, index=False)
    updated.to_csv(POST_DRIFT_PATH, index=False)

    print(f"[SUCCESS] Updated training dataset saved to {OUTPUT_PATH} ({len(updated)} rows)")


if __name__ == "__main__":
    main()

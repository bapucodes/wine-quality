import pandas as pd
from pathlib import Path

SRC_PATH = Path("data/raw/winequality.csv")
DEST_PATH = Path("data/monitoring/drift/evolved_data.csv")


def main():
    print("[INFO] Creating evolved dataset for drift simulation...")
    df = pd.read_csv(SRC_PATH)
    evolved = df.copy()

    # Apply realistic seasonal chemical drift
    evolved["alcohol"] = (evolved["alcohol"] * 1.15).round(2)
    evolved["volatile acidity"] = (evolved["volatile acidity"] * 1.25).round(3)
    evolved["sulphates"] = (evolved["sulphates"] * 0.85).round(2)

    if "quality" in evolved.columns:
        evolved = evolved.drop(columns=["quality"])

    DEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    evolved.to_csv(DEST_PATH, index=False)
    print(f"[SUCCESS] Evolved dataset created at {DEST_PATH} ({len(evolved)} rows)")


if __name__ == "__main__":
    main()

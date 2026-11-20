import pandas as pd
from pathlib import Path

SRC_PATH = Path("data/raw/winequality.csv")
DEST_PATH = Path("data/monitoring/drift/evolved_labeled_data.csv")


def main():
    print("[INFO] Creating labeled concept drift dataset...")
    df = pd.read_csv(SRC_PATH)
    evolved = df.copy()

    # Shift feature values
    evolved["alcohol"] = (evolved["alcohol"] * 1.15).round(2)
    evolved["volatile acidity"] = (evolved["volatile acidity"] * 1.25).round(3)
    evolved["sulphates"] = (evolved["sulphates"] * 0.85).round(2)

    # Shift target concept labels slightly
    def shift_label(row):
        score = row["quality"]
        if row["volatile acidity"] > 0.7:
            return max(3, score - 1)
        if row["alcohol"] > 11.5 and row["sulphates"] > 0.65:
            return min(8, score + 1)
        return score

    evolved["quality"] = evolved.apply(shift_label, axis=1)

    DEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    evolved.to_csv(DEST_PATH, index=False)
    print(f"[SUCCESS] Concept drift labeled dataset created at {DEST_PATH}")


if __name__ == "__main__":
    main()

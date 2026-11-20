import json
from pathlib import Path

LOG_PATH = Path("logs/inference/inference_log.jsonl")


def main():
    print("[INFO] Monitoring prediction distributions...")
    total = 0
    good_wines = 0
    if LOG_PATH.exists():
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    total += 1
                    if entry.get("prediction") == 1:
                        good_wines += 1
                except Exception:
                    continue

    print(f"[INFO] Total logged inference requests: {total}")
    if total > 0:
        print(f"[INFO] Predicted High Quality Wine: {good_wines} ({(good_wines/total)*100:.1f}%)")
    print("[SUCCESS] Prediction distribution monitoring complete.")


if __name__ == "__main__":
    main()

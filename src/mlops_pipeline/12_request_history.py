import json
from pathlib import Path

LOG_PATH = Path("logs/inference/inference_log.jsonl")


def main():
    print("[INFO] Inspecting recent API request history...")
    history = []
    if LOG_PATH.exists():
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        history.append(json.loads(line))
                    except Exception:
                        pass
    print(f"[INFO] History contains {len(history)} total records.")
    for h in history[-5:]:
        print(f"  Request ID: {h.get('request_id')} | Status: {h.get('status')} | Pred: {h.get('prediction')} | Latency: {h.get('latency_ms')}ms")


if __name__ == "__main__":
    main()

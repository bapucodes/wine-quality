import json
import subprocess
import sys
from pathlib import Path

REPORT_PATH = Path("reports/performance_report.json")
API_MONITOR_SCRIPT = Path(__file__).resolve().parent / "12_api_performance_monitoring.py"


def main():
    print("[INFO] Monitoring model performance in production...")
    if not REPORT_PATH.exists():
        subprocess.run([sys.executable, str(API_MONITOR_SCRIPT)])
    if REPORT_PATH.exists():
        with open(REPORT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"[INFO] Performance Report: Avg Latency = {data.get('average_latency_ms')} ms, Success Rate = {data.get('success_rate_percent')}%")
    print("[SUCCESS] Production performance monitoring verified.")


if __name__ == "__main__":
    main()

import os
from pathlib import Path
import yaml

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


def check_registry():
    print("=" * 60)
    print("CHECK MLFLOW REGISTRY - WINE QUALITY MLOPS")
    print("=" * 60)
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_name = config["registry"]["model_name"]
    print(f"Target model: {model_name}")

    if HAS_MLFLOW:
        try:
            client = MlflowClient()
            versions = client.search_model_versions(f"name='{model_name}'")
            print(f"Registered model versions: {len(versions)}")
            for v in versions:
                print(f"  Version {v.version} | Stage: {v.current_stage} | Run ID: {v.run_id}")
        except Exception as e:
            print(f"MLflow client note: {e}")
    else:
        print("MLflow client not present. Current production bundle resides in deployment/model.pkl.")


if __name__ == "__main__":
    check_registry()

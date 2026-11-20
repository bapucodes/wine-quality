import json
import shutil
from pathlib import Path
import joblib
import yaml

try:
    import mlflow
    import mlflow.sklearn
    from mlflow import MlflowClient
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

MODEL_NAME = "Wine_Quality_Production_Model"


def load_config():
    project_root = Path(__file__).resolve().parents[2]
    with open(project_root / "config.yaml", "r", encoding="utf-8") as file:
        return project_root, yaml.safe_load(file)


def prepare_deployment():
    project_root, config = load_config()
    deployment_dir = project_root / "deployment"
    deployment_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Preparing Production deployment bundle - Wine Quality MLOps...")

    model_src = project_root / config["paths"]["model"]
    preprocessor_src = project_root / config["paths"]["preprocessor"]

    if not model_src.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_src}")
    if not preprocessor_src.exists():
        raise FileNotFoundError(f"Preprocessor artifact not found at {preprocessor_src}")

    model_dest = deployment_dir / "model.pkl"
    preprocessor_dest = deployment_dir / "preprocessor.pkl"

    shutil.copy2(model_src, model_dest)
    shutil.copy2(preprocessor_src, preprocessor_dest)

    # Verification
    loaded_model = joblib.load(model_dest)
    loaded_preprocessor = joblib.load(preprocessor_dest)

    if loaded_model is None or loaded_preprocessor is None:
        raise RuntimeError("Deployment artifact verification failed.")

    metadata = {
        "model_name": MODEL_NAME,
        "model_version": "1",
        "model_stage": "Production",
        "run_id": "run_wine_quality_local_01",
        "model_source": str(model_dest).replace("\\", "/"),
        "model_id": "wine_quality_rf_v1",
        "preprocessor_artifact": "preprocessing_pipeline/preprocessor.pkl",
        "source": "Local ML Pipeline",
    }

    metadata_path = deployment_dir / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print(f"[SUCCESS] Model saved to {model_dest}")
    print(f"[SUCCESS] Preprocessor saved to {preprocessor_dest}")
    print(f"[SUCCESS] Metadata saved to {metadata_path}")
    print("\n[INFO] Deployment bundle ready:")
    print("deployment/")
    print("├── model.pkl")
    print("├── preprocessor.pkl")
    print("└── model_metadata.json")
    print("\n[SUCCESS] Deployment preparation completed.")


if __name__ == "__main__":
    prepare_deployment()

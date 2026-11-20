import sys
from pathlib import Path

# Delegate to mlops_pipeline
pipeline_train = Path(__file__).resolve().parent / "mlops_pipeline" / "05_train_registry.py"
with open(pipeline_train, "r", encoding="utf-8") as f:
    code = f.read()
exec(code)

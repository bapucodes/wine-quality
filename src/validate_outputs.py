import sys
from pathlib import Path

# Delegate to mlops_pipeline implementation
pipeline_validate = Path(__file__).resolve().parent / "mlops_pipeline" / "04_validate_outputs.py"
with open(pipeline_validate, "r", encoding="utf-8") as f:
    code = f.read()
exec(code)

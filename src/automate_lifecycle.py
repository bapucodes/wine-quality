import sys
from pathlib import Path

# Delegate to mlops_pipeline
pipeline_lifecycle = Path(__file__).resolve().parent / "mlops_pipeline" / "08_automate_lifecycle.py"
with open(pipeline_lifecycle, "r", encoding="utf-8") as f:
    code = f.read()
exec(code)

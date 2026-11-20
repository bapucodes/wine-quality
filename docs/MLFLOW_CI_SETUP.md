# MLflow Tracking for Labs 7–9

The project reads the MLflow Tracking URI from:

1. `MLFLOW_TRACKING_URI` environment variable, or
2. `mlflow.tracking_uri` in `config.yaml`.

For local development without an external tracking server, the system provides automatic local artifact logging and mocking fallback.

To connect to a persistent MLflow server:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 127.0.0.1 --port 5000
```

PowerShell:
```powershell
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
```

Linux/macOS:
```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

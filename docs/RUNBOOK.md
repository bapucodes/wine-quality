# Runbook — Red Wine Quality Prediction MLOps

## 1. Prerequisites & Environment Setup

1. Confirm Python 3.11+ is installed.
2. Install project dependencies:
   ```powershell
   python -m pip install -r requirements.txt
   ```
3. Verify directory structure and configuration:
   ```powershell
   Get-ChildItem -Directory
   Get-Content config.yaml
   ```

---

## 2. Running Pipeline Stages Locally

### Execute End-to-End MLOps Pipeline:
```powershell
python src/mlops_pipeline/wine_quality_pipeline.py --force
```

### Execute Baseline Pipeline (Lab 3):
```powershell
python pipelines/run_lab3_baseline.py
```

### Execute Experiment Tracking & Reproducibility (Lab 4):
```powershell
python pipelines/run_lab4_tracking.py
```

### Execute Production Preprocessing Pipeline (Lab 5):
```powershell
python pipelines/run_lab5_pipeline.py
```

### Execute Model Registry & Lifecycle Promotion (Lab 6):
```powershell
python pipelines/run_lab6_registry.py
```

---

## 3. Running Failure Injection & Recovery Tests (Lab 9)

Run the 5 failure scenarios across 3 repetitions:
```powershell
python src/mlops_pipeline/10_failure_injection.py
```
Inspect the resulting report at `reports/failure_report.json`.

---

## 4. Serving & API Testing (Lab 10)

### Start FastAPI Server:
```powershell
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# or:
python run_api.py
```

### Execute API & Performance Test Suite:
```powershell
python -m pytest tests/test_api.py tests/test_performance.py -v
```

---

## 5. Drift Analysis & Retraining Triggers (Lab 12 & 13)

### Run Drift Detection:
```powershell
python src/mlops_pipeline/12_data_drift.py
python src/mlops_pipeline/12_prediction_drift.py
python src/mlops_pipeline/12_concept_drift.py
python src/mlops_pipeline/12_drift_analysis.py
```

### Run Retraining Trigger:
```powershell
python src/mlops_pipeline/12_monitor_trigger.py
```

---

## 6. Container & Kubernetes Operations (Lab 11)

### Docker Build & Run:
```powershell
docker build -t wine-quality-api:1.0 .
docker run -d -p 8000:8000 --name wine-api wine-quality-api:1.0
```

### Test Local vs Container Consistency:
```powershell
python tests/test_local_vs_container.py
```

### Deploy to Kubernetes:
```powershell
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl get pods -l app=wine-quality-api
kubectl get svc wine-quality-api-service
```

---

## 7. Troubleshooting

- **Dataset Re-ingestion Skipped**: If `data/raw/archive/dataset_lineage.json` contains the SHA-256 hash of `new_winequality.csv`, the pipeline detects `already_ingested` and skips retraining. Use `--force` on `wine_quality_pipeline.py` or introduce a new batch hash to retrain.
- **Model Promotion Rejected**: If a candidate model has recall lower than the active Production model, it will be safely rejected and archived. Check `reports/lifecycle_report.json`.

# MLOps Red Wine Quality Prediction — Architecture

## Purpose

This repository implements a production-grade Red Wine Quality prediction platform with schema validation, preprocessing, experiment/model tracking, a recall quality gate, model lifecycle decisions, monitoring-triggered retraining, and Docker/Kubernetes model serving.

## End-to-End Flow

```text
GitHub Actions / Local Scheduler
              |
              v
12_monitor_trigger.py
   |          |
   |          +--> Read existing drift reports
   |               - data drift (KS test)
   |               - prediction drift (KS test)
   |               - concept/performance drift
   |
   +--> Check for data/raw/new_winequality.csv
              |
              v
wine_quality_pipeline.py (when drift and a new batch are present)
              |
              v
01 Data Loading + SHA-256 batch hash/lineage checks
              |
              v
02 Raw validation --> 03 Preprocessing --> 04 Output validation
              |
              v
05 Training + MLflow / Model Registry
              |
              v
06 Evaluation --> 07 Recall quality gate (minimum 0.70)
              |
              v
08 Automated lifecycle decision (Candidate vs Production)
              |
        +-----+----------------+
        |                      |
   Not promoted             Promoted
        |                      |
        v                      v
Skip deployment        Prepare Production model bundle (deployment/)
                               |
                               v
                       Docker image + Kubernetes rollout
              |
              v
09 Production inference
```

## Main Components

| Component | Responsibility |
|---|---|
| `src/mlops_pipeline/12_monitor_trigger.py` | Runs drift analysis, checks for an incoming batch, calls the main pipeline when appropriate, and writes a monitoring-trigger report. |
| `src/mlops_pipeline/12_drift_analysis.py` | Combines existing data-, prediction-, and concept-drift reports into an overall summary. |
| `src/mlops_pipeline/01_load_data.py` | Validates incoming schema, hashes the batch with SHA-256, prevents duplicate ingestion, archives prior datasets, appends new rows, and records lineage. |
| `src/mlops_pipeline/wine_quality_pipeline.py` | Runs ordered ML stages 01 through 09 and stops safely if the incoming batch was already ingested. |
| `src/mlops_pipeline/05_train_registry.py` | Trains a candidate Random Forest model, logs metrics/params, and registers the model in the registry. |
| `src/mlops_pipeline/07_quality_gate.py` | Requires recall to meet the configured minimum of 0.70 (achieved: 0.7670). Passing this gate alone does not promote a model. |
| `src/mlops_pipeline/08_automate_lifecycle.py` | Compares the candidate with Production and records promotion/archive decisions in `reports/lifecycle_report.json`. |
| `src/mlops_pipeline/deploy_if_promoted.py` | Invokes deployment only when the lifecycle report says the candidate was promoted. |
| `src/mlops_pipeline/11_prepare_deployment.py` | Packages the registered Production model, its matching preprocessor, and metadata into `deployment/`. |
| `deployment/deploy_production.py` | Builds or reuses a model Docker image, updates Kubernetes, waits for rollout, and writes a deployment report. |
| `api/main.py` | FastAPI inference application serving `/health`, `/metadata`, `/predict`, and `/predict/batch`. |
| `.github/workflows/` | CI/CD automation workflows for pipeline execution, failure injection, and monitoring. |

## Data Ingestion & Idempotency

`data/raw/new_winequality.csv` being present does not prove that a batch is new. The loader computes a SHA-256 hash and compares it with `data/raw/archive/dataset_lineage.json`. If the hash already exists, it reports `already_ingested` and skips retraining. For a new hash, it validates schema and sample IDs (`wine_id`), archives the previous dataset version, appends the batch to `data/raw/winequality.csv`, and updates lineage.

# Red Wine Quality Prediction — End-to-End MLOps Platform

An enterprise-grade, production-ready Machine Learning Operations (MLOps) platform for predicting Red Wine Quality from physicochemical characteristics, demonstrating end-to-end data lineage, schema validation, experiment tracking, automated quality gating, model lifecycle management, containerized serving with Kubernetes, multi-modal drift monitoring, and self-healing resilience.

**Project Status:** Labs 1–13 fully integrated and operational.  
**Test Suite:** 18/18 Automated Pytest Unit & Performance Tests Passed ($100\%$ pass rate).  
**Resilience Suite:** 5 Failure Injection Scenarios $\times$ 3 Repetitions Passed ($100\%$ safe-halt/recovery).  
**Quality Gate:** Recall threshold $\ge 0.70$ satisfied (Achieved: $0.7670$, ROC-AUC: $0.8892$).  

---

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Key Capabilities](#key-capabilities)
- [Architecture](#architecture)
- [End-to-End Lifecycle](#end-to-end-lifecycle)
- [Dataset & Schema Definition](#dataset--schema-definition)
- [Model Evaluation & Quality Gate Policy](#model-evaluation--quality-gate-policy)
- [Technology Stack](#technology-stack)
- [Repository Layout](#repository-layout)
- [Quickstart & Environment Setup](#quickstart--environment-setup)
- [Running the Pipelines](#running-the-pipelines)
- [FastAPI Model Serving](#fastapi-model-serving)
- [Failure Injection & Self-Healing Testing](#failure-injection--self-healing-testing)
- [Drift Monitoring & Retraining Triggers](#drift-monitoring--retraining-triggers)
- [Containerization & Kubernetes Deployment](#containerization--kubernetes-deployment)
- [CI/CD Automation](#cicd-automation)
- [Artifacts & Reports Summary](#artifacts--reports-summary)

---

## Project Overview

The **Red Wine Quality MLOps Platform** models red wine samples based on 11 physicochemical properties (fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol) to predict whether a given vintage is of **Good Quality** (Class 1: rating $\ge 6$) or **Standard Quality** (Class 0: rating $< 6$).

A central design tenet of this system is that **a trained model is never automatically deployed to production**. A model candidate must satisfy strict data validation, achieve deterministic reproducibility, pass the recall quality gate, outperform the current production champion, and package atomic artifacts before reaching deployment.

---

## Objectives

- **Automated Data Validation**: Prevent schema corruption, missing features, and anomalous ranges using Pandera and SHA-256 lineage tracking.
- **Reproducible ML Pipeline**: Ensure identical metrics and artifact generation across isolated runs.
- **Strict Quality Gating**: Enforce minimum recall ($\ge 0.70$) to prevent inferior models from reaching the registry.
- **Automated Model Lifecycle Promotion**: Champion vs. Challenger evaluation that automatically promotes or archives candidate models.
- **Low-Latency Model Serving**: FastAPI microservice delivering sub-15ms inference latencies with 100% reliability.
- **Failure Resilience**: Automatic detection and recovery when model binaries, schemas, or raw files are corrupted.
- **Continuous Monitoring**: Multi-modal drift detection (Kolmogorov-Smirnov data drift, prediction drift, and concept drift) driving automated retraining.
- **Cloud-Native Deployment**: Production-ready Docker containerization and Kubernetes manifests (Deployment, ReplicaSets, NodePort Service).

---

## Key Capabilities

| Capability | Module | Description |
|---|---|---|
| **Data Lineage Tracking** | `01_load_data.py` | SHA-256 file hashing, duplicate batch rejection, and historical archiving |
| **Schema Validation** | `02_validate_data.py` | Pandera-driven schema contracts checking column existence, types, and bounds |
| **Output Validation** | `04_validate_outputs.py` | Validates post-scaling matrices for NaNs, infinites, and correct dimensions |
| **Experiment Tracking** | `05_train_registry.py` | Automated metric logging, hyperparameter tracking, and model registration |
| **Quality Gate** | `07_quality_gate.py` | Production gate requiring recall $\ge 0.70$ before promotion consideration |
| **Lifecycle Promotion** | `08_automate_lifecycle.py` | Compares candidate recall vs. active Production model to decide promotion |
| **Offline Inference** | `09_inference.py` | Batch inference pipeline saving predictions to `outputs/inference_predictions.csv` |
| **Fault Injection & Recovery** | `10_failure_injection.py` | Injects 5 fatal failure modes across 3 repetitions; verifies HALT and RECOVER |
| **FastAPI Serving** | `api/main.py` | High-throughput REST API with `/health`, `/metadata`, `/predict`, `/predict/batch` |
| **Data Drift Detection** | `12_data_drift.py` | Two-sample Kolmogorov-Smirnov test tracking shifts across feature distributions |
| **Prediction Drift** | `12_prediction_drift.py` | Detects shifts in model output classification and probability distributions |
| **Concept Drift** | `12_concept_drift.py` | Evaluates performance degradation on incoming evolved ground-truth data |
| **Retraining Trigger** | `12_monitor_trigger.py` | Automated trigger that kicks off retraining only when drift & new data align |

---

## Architecture

```text
Incoming Dataset Batch (new_winequality.csv)
                    |
                    v
       [01 Data Lineage & SHA-256 Hash]
        |                           |
    (New Hash)                 (Duplicate)
        |                           |
        v                           v
[02 Schema Validation]          HALT (Idempotent)
        |
        v
[03 Preprocessing & Scaling]
        |
        v
[04 Processed Output Validation]
        |
        v
[05 Model Training & Registry]
        |
        v
[06 Model Evaluation (ROC-AUC / Confusion Matrix)]
        |
        v
[07 Quality Gate (Recall >= 0.70)]
        |
   (Passed Gate)
        |
        v
[08 Automated Lifecycle Promotion (Candidate vs Production)]
        |                                       |
    (Promoted)                             (Rejected)
        |                                       |
        v                                       v
[11 Prepare Deployment Bundle]             Archive Candidate
        |
        +---------------------------------------+
        |                                       |
        v                                       v
[Docker Containerization]               [Kubernetes Rollout]
        |                                       |
        +---------------------------------------+
        |
        v
[FastAPI Serving (/health, /metadata, /predict, /predict/batch)]
        |
        v
[Continuous Drift Monitoring (Data, Prediction, Concept)]
        |
        v
[Automated Retraining Trigger (12_monitor_trigger.py)]
```

---

## Dataset & Schema Definition

The dataset consists of 1,649 total samples across 11 physicochemical laboratory metrics:

| Column | Type | Valid Range | Physical Meaning |
|---|---|---|---|
| `wine_id` | String | `WINE_XXXX` | Unique sample identifier for test tracking |
| `fixed acidity` | Float | $[3.0, 20.0]$ | Tartaric acid concentration ($g/dm^3$) |
| `volatile acidity` | Float | $[0.0, 3.0]$ | Acetic acid concentration ($g/dm^3$) |
| `citric acid` | Float | $[0.0, 2.0]$ | Citric acid concentration ($g/dm^3$) |
| `residual sugar` | Float | $[0.0, 50.0]$ | Remaining sugar after fermentation ($g/dm^3$) |
| `chlorides` | Float | $[0.0, 1.0]$ | Salt concentration ($g/dm^3$) |
| `free sulfur dioxide` | Float | $[0.0, 100.0]$ | Free $SO_2$ preventing microbial growth ($mg/dm^3$) |
| `total sulfur dioxide` | Float | $[0.0, 400.0]$ | Bound + free $SO_2$ concentration ($mg/dm^3$) |
| `density` | Float | $[0.90, 1.10]$ | Density relative to water ($g/cm^3$) |
| `pH` | Float | $[2.0, 5.0]$ | Solution acidity / alkalinity scale |
| `sulphates` | Float | $[0.0, 3.0]$ | Potassium sulphate additive ($g/dm^3$) |
| `alcohol` | Float | $[5.0, 25.0]$ | Ethanol volume percent ($\%$ vol) |
| `quality` | Integer | $[3, 8]$ | Expert sensory rating |

**Target Transformation:**
$$\text{Class} = \begin{cases} 1 & \text{if } \text{quality} \ge 6 \quad (\text{Good Quality}) \\ 0 & \text{if } \text{quality} < 6 \quad (\text{Standard Quality}) \end{cases}$$

---

## Model Evaluation & Quality Gate Policy

### Champion Model Metrics (Random Forest):
- **Accuracy**: $0.7879$ ($78.79\%$)
- **Precision**: $0.8232$ ($82.32\%$)
- **Recall**: $0.7670$ ($76.70\%$)
- **F1 Score**: $0.7941$ ($79.41\%$)
- **ROC-AUC**: $0.8892$

### Quality Gate Policy:
```yaml
quality_gate:
  metric: recall
  minimum_value: 0.70
```
- **Evaluation**: The candidate model achieved a recall of **0.7670**, successfully passing the 0.70 minimum threshold.
- **Promotion Rule**: Even if a candidate passes the quality gate, it is only promoted if its primary metric exceeds the currently active Production model version.

---

## Technology Stack

- **ML Core**: Python 3.11+, Scikit-Learn, NumPy, Pandas, SciPy, Joblib
- **Data Contract & Validation**: Pandera, SHA-256 Hashing
- **Serving & API**: FastAPI, Starlette, Uvicorn, Pydantic v2
- **Testing**: Pytest, Starlette TestClient, Requests
- **Containerization & Orchestration**: Docker, Kubernetes (Deployment, Service)
- **CI/CD Automation**: GitHub Actions Workflows

---

## Repository Layout

```text
.
|-- .dockerignore
|-- .github/workflows/
|   |-- lab8_ci.yml
|   |-- lab9_failure_tests.yml
|   |-- lab13_monitoring.yml
|   \-- lab13_runner_test.yml
|-- api/
|   |-- __init__.py
|   |-- main.py
|   |-- model_loader.py
|   |-- monitoring_logger.py
|   |-- preprocessing.py
|   |-- schemas.py
|   |-- stage9_flow.txt
|   \-- stages.txt
|-- artifacts/
|   |-- api_performance_report.json
|   |-- baseline_validation.csv
|   |-- confusion_matrix.png
|   |-- corrupted_simulation_errors.csv
|   |-- data_load_report.json
|   |-- pipeline_reproducibility_report.json
|   |-- preprocessing_summary_report.json
|   |-- production_model_report.json
|   |-- roc_curve.png
|   \-- scaler.pkl
|-- config.yaml
|-- data/
|   |-- monitoring/drift/
|   |   |-- evolved_data.csv
|   |   |-- evolved_labeled_data.csv
|   |   \-- post_drift_training_data.csv
|   |-- processed/
|   |   |-- dataset_metadata.json
|   |   |-- test_wine_ids.csv
|   |   |-- updated_training_data.csv
|   |   |-- X_train_final.npy, X_test_final.npy
|   |   \-- y_train.npy, y_test.npy
|   \-- raw/
|       |-- archive/ (version archives & dataset_lineage.json)
|       |-- winequality.csv
|       \-- new_winequality.csv
|-- deployment/
|   |-- deploy_production.py
|   |-- model.pkl
|   |-- model_metadata.json
|   \-- preprocessor.pkl
|-- Dockerfile
|-- docs/
|   |-- ARCHITECTURE.md
|   |-- RUNBOOK.md
|   |-- LAB7_PIPELINE.md
|   |-- LAB8_CI_REPORT.md
|   |-- LAB9_FAILURE_HANDLING_REPORT.md
|   |-- MLFLOW_CI_SETUP.md
|   |-- LAB13_END_TO_END.md
|   |-- lab11_containerization_report.md
|   \-- lab11_deployment_debugging_report.md
|-- k8s/
|   |-- deployment.yaml
|   \-- service.yaml
|-- logs/
|   \-- inference/inference_log.jsonl
|-- models/
|   |-- retrained_model_v2.pkl
|   \-- wine_quality_model.pkl
|-- notebooks/
|   |-- wine_quality_prediction.ipynb
|   \-- project_implementation.ipynb
|-- outputs/
|   |-- drift/
|   |   |-- data_drift_report.csv
|   |   |-- prediction_drift_report.csv
|   |   |-- concept_drift_report.csv
|   |   \-- drift_analysis_report.csv
|   |-- inference_predictions.csv
|   \-- monitoring/
|       |-- monitor_trigger_report.json
|       \-- performance_trend.json
|-- pipelines/
|   |-- run_lab3_baseline.py
|   |-- run_lab4_tracking.py
|   |-- run_lab5_pipeline.py
|   \-- run_lab6_registry.py
|-- reports/
|   |-- api_testing_report.json
|   |-- deployment_automation_report.json
|   |-- evaluation_report.json
|   |-- failure_report.json
|   |-- lab10_summary.md
|   |-- lifecycle_report.json
|   |-- performance_report.json
|   \-- quality_gate_report.json
|-- src/
|   |-- mlops_pipeline/ (01 through 12 scripts)
|   |-- preprocess.py, train.py, evaluate.py
|   \-- validate_data.py, validate_outputs.py, validate_reproducibility.py
\-- tests/
    |-- test_api.py
    |-- test_performance.py
    \-- test_local_vs_container.py
```

---

## Quickstart & Environment Setup

### 1. Clone & Set Up Python Virtual Environment:
```bash
git clone https://github.com/user/wine-quality-mlops.git
cd wine-quality-mlops
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies:
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## Running the Pipelines

### Execute the Unified MLOps Pipeline:
```bash
python src/mlops_pipeline/wine_quality_pipeline.py --force
```

### Execute Staged Lab Pipelines:
```bash
# Lab 3: Baseline preprocessing, training, and evaluation
python pipelines/run_lab3_baseline.py

# Lab 4: Experiment tracking & 100% reproducibility validation
python pipelines/run_lab4_tracking.py

# Lab 5: Production data validation & output sanity checks
python pipelines/run_lab5_pipeline.py

# Lab 6: Model registry lifecycle management & reporting
python pipelines/run_lab6_registry.py
```

---

## FastAPI Model Serving

### 1. Start Inference Server:
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# or simply:
python run_api.py
```
Interactive API docs available at: `http://localhost:8000/docs`

### 2. Single Sample Prediction (`POST /predict`):
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "wine_id": "WINE_SAMPLE",
       "fixed acidity": 7.4,
       "volatile acidity": 0.70,
       "citric acid": 0.00,
       "residual sugar": 1.9,
       "chlorides": 0.076,
       "free sulfur dioxide": 11.0,
       "total sulfur dioxide": 34.0,
       "density": 0.9978,
       "pH": 3.51,
       "sulphates": 0.56,
       "alcohol": 9.4
     }'
```
**Response:**
```json
{
  "prediction": 0,
  "quality_probability": 0.0756,
  "churn_probability": 0.0756
}
```

### 3. Run Automated API & Performance Test Suite:
```bash
python -m pytest tests/test_api.py tests/test_performance.py -v
```
Result: **18 passed in 4.52s** (100% pass rate).

---

## Failure Injection & Self-Healing Testing

The system evaluates 5 failure scenarios across 3 repetitions to ensure safe-fail and self-healing:
```bash
python src/mlops_pipeline/10_failure_injection.py
```

| Scenario | Tested Behavior | Actual Outcome | Status |
|---|---|---|---|
| **1. Missing Raw Data** | Pipeline halts before execution | HALT | PASSED |
| **2. Invalid Datatype** | Schema validation detects non-float strings | HALT | PASSED |
| **3. Missing Feature** | Schema validation detects dropped column | HALT | PASSED |
| **4. Output NaN Injection** | Processed output validation catches NaNs | HALT | PASSED |
| **5. Missing Model Binary** | Deletes `models/wine_quality_model.pkl` | Self-healed & recovered from Registry | PASSED |

Full evidence: `reports/failure_report.json`.

---

## Drift Monitoring & Retraining Triggers

### 1. Run Drift Detection:
```bash
# Feature distribution data drift (Kolmogorov-Smirnov test)
python src/mlops_pipeline/12_data_drift.py

# Prediction & probability output drift
python src/mlops_pipeline/12_prediction_drift.py

# Ground truth concept & performance degradation drift
python src/mlops_pipeline/12_concept_drift.py

# Combined drift analysis summary
python src/mlops_pipeline/12_drift_analysis.py
```

### 2. Retraining Trigger Execution:
```bash
python src/mlops_pipeline/12_monitor_trigger.py
```
- When drift is detected AND an unconsumed dataset batch (`new_winequality.csv`) is present, the pipeline triggers automated retraining, generates Model V2, and records a comparison report in `outputs/retraining/model_comparison_report.json`.
- When an already-ingested batch is detected, the pipeline halts idempotently without redundant training.

---

## Containerization & Kubernetes Deployment

### 1. Build & Run Docker Container:
```bash
docker build -t wine-quality-api:1.0 .
docker run -d -p 8000:8000 --name wine-api wine-quality-api:1.0
```

### 2. Verify Local vs. Container Prediction Consistency:
```bash
python tests/test_local_vs_container.py
```

### 3. Deploy onto Kubernetes:
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

kubectl get deployments
kubectl get pods -l app=wine-quality-api
kubectl get svc wine-quality-api-service
```

---

## CI/CD Automation

- **`lab8_ci.yml`**: Automatically triggers on pushes or pull requests touching code, config, or raw data, executing data validation, model training, evaluation, quality gating, and pytest suites.
- **`lab9_failure_tests.yml`**: Manual and automated workflow running the 5 failure injection scenarios across 3 repetitions.
- **`lab13_monitoring.yml`**: Scheduled cron workflow running drift monitoring and retraining triggers against incoming batches.

---

## Artifacts & Reports Summary

All verification and run metrics are recorded in standardized JSON and CSV artifacts:
- `reports/evaluation_report.json`: Classification metrics & confusion matrix
- `reports/quality_gate_report.json`: Recall $\ge 0.70$ quality threshold verification
- `reports/lifecycle_report.json`: Model candidate vs. Production promotion decision
- `reports/failure_report.json`: 15/15 passing failure injection tests
- `reports/api_testing_report.json`: API functionality and edge-case test results
- `reports/performance_report.json`: Latency (9.86ms avg), throughput (101 req/s), and reliability (100%)
- `artifacts/confusion_matrix.png` & `artifacts/roc_curve.png`: Model visual performance
- `outputs/drift/drift_analysis_report.csv`: Combined multi-modal drift report

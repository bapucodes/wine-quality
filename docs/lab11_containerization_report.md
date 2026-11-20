# Lab 11 — Containerization and Kubernetes Deployment Report

## 1. Objective

The objective of Lab 11 is to containerize the Red Wine Quality Prediction FastAPI service, validate container-local parity, deploy the containerized application onto Kubernetes, demonstrate pod scaling, simulate deployment failures, and validate self-healing behavior.

---

## 2. Deployment Architecture

```text
Model Registry
     ↓
Production Model Version
     ↓
Matching Model + Preprocessor
     ↓
Deployment Bundle (deployment/)
     ↓
Docker Image (wine-quality-api:1.0)
     ↓
Kubernetes Deployment & ReplicaSet
     ↓
Kubernetes Pods & Service (:8000)
     ↓
FastAPI Serving
```

---

## 3. Production Model & Verification

- **Model Name**: `Wine_Quality_Production_Model`
- **Version**: 1
- **Latency**: $9.86$ ms
- **Reliability**: $100\%$
- **Test Suite Result**: 18/18 tests passed

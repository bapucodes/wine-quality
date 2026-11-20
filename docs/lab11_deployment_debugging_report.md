# Lab 11 — Deployment and Debugging Report

## 1. Objective

The objective of this activity is to simulate deployment failures, identify root causes using container logs, and validate Kubernetes self-healing and service resilience.

---

## 2. Simulated Scenarios & Recovery

### Scenario 1: Missing Model Artifact in Container
- **Symptom**: Application fails readiness probe or raises 503 Service Unavailable.
- **Root Cause**: Deployment bundle omitted `model.pkl`.
- **Resolution**: `11_prepare_deployment.py` automated bundling ensures atomic packaging of model, preprocessor, and metadata.

### Scenario 2: Pod Termination & Self-Healing
- **Action**: Pod killed using `kubectl delete pod <pod-name>`.
- **Behavior**: Kubernetes ReplicaSet immediately detects pod absence and launches a replacement pod in under 5 seconds.
- **Service Continuity**: Zero downtime observed via NodePort service routing.

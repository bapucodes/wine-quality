# Lab 11 - Containerization and Deployment of ML Systems

## Purpose
This document provides the complete roadmap, verification procedure, and deployment specification for the Red Wine Quality MLOps Platform.

---

# 1. Starting Point & Architecture
- **Service**: FastAPI Serving Application
- **Endpoints**:
  - `GET /`: Health ping & system info
  - `GET /health`: Liveness probe
  - `GET /metadata`: Serving model version, schema, and readiness
  - `POST /predict`: Single sample inference
  - `POST /predict/batch`: High-throughput batch inference
- **Quality Gates**: Recall $\ge 0.70$
- **Performance Benchmark**: Single request latency $< 25$ ms, 100% reliability

---

# 2. Containerization Lifecycle
1. **Model Staging**: `src/mlops_pipeline/11_prepare_deployment.py` extracts verified artifacts from the Model Registry into `deployment/`.
2. **Docker Build**:
   ```bash
   docker build -t wine-quality-api:1.0 .
   ```
3. **Local Container Execution**:
   ```bash
   docker run -d -p 8000:8000 --name wine-api wine-quality-api:1.0
   ```
4. **Parity Verification**:
   ```bash
   python tests/test_local_vs_container.py
   ```

---

# 3. Kubernetes Orchestration
1. **Cluster Verification**:
   ```bash
   kubectl cluster-info
   kubectl get nodes
   ```
2. **Deployment Application**:
   ```bash
   kubectl apply -f k8s/deployment.yaml
   kubectl apply -f k8s/service.yaml
   ```
3. **Pod & Service Status**:
   ```bash
   kubectl get deployments
   kubectl get pods -l app=wine-quality-api
   kubectl get svc wine-quality-api-service
   ```
4. **Horizontal Scaling**:
   ```bash
   kubectl scale deployment/wine-quality-api --replicas=3
   ```

---

# 4. Self-Healing & Failure Recovery
- **Pod Deletion Test**: When a pod instance is terminated (`kubectl delete pod <pod-name>`), the ReplicaSet automatically re-spawns a replacement pod within seconds to maintain high availability.
- **NodePort Forwarding**: The service binds port 8000 to the container target port 8000, ensuring seamless routing.

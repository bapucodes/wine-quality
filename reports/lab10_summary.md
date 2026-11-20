# Lab 10 - ML Model Serving and API Engineering using FastAPI

## 1. Objective

The objective of Lab 10 is to deploy the trained machine learning model as a FastAPI inference service and evaluate its API functionality, validation behavior, robustness, performance, and reliability for the Red Wine Quality MLOps Platform.

---

## 2. Implementation Completed

The following components were implemented and verified:

- **FastAPI Application**: High-performance RESTful API with lifespan lifecycle management.
- **Health Endpoint** (`GET /health`): Probes API liveness and container operational readiness.
- **ModelLoader**: Automated loading of serialized production model, preprocessor, and metadata from `deployment/`.
- **Metadata Endpoint** (`GET /metadata`): Returns service name, version, model URI, model loading state, and feature schema.
- **Single Prediction Endpoint** (`POST /predict`): Real-time inference returning binary prediction (`0: standard`, `1: good`) and quality probability.
- **Batch Prediction Endpoint** (`POST /predict/batch`): High-throughput vectorized inference supporting bulk evaluation.
- **Pydantic Validation**: Strict numerical bounds (`ge=0`), type checking, and alias conversion.
- **Standardized Error Handling**: JSON 422 Unprocessable Entity responses for missing fields, invalid types, and out-of-range inputs.
- **Inference Logging**: JSONL structured append log (`logs/inference/inference_log.jsonl`) recording timestamps, request IDs, inputs, predictions, probabilities, and latencies.
- **Automated Test Suite**: 18/18 passing pytest test cases for functionality, validation, latency, and reliability.

---

## 3. API Endpoints

### `GET /`
Returns service status and lab identifier.

### `GET /health`
Liveness check returning `{"status": "healthy"}`.

### `GET /metadata`
Returns service version, active model name, model URI, and readiness flags.

### `POST /predict`
Evaluates a single wine sample across 11 physicochemical properties.
Returns:
- `prediction`: `0` or `1`
- `quality_probability`: confidence score $[0.0, 1.0]$

### `POST /predict/batch`
Accepts a batch of wine samples and returns array of predictions with total sample count.

---

## 4. Performance & Reliability Results

- **Single Request Latency**: $12.45$ ms
- **Average Latency**: $9.86$ ms
- **Median Latency**: $9.21$ ms
- **Throughput**: $101.42$ requests/sec
- **Success Rate**: $100.0\%$ (0 failures across all runs)
- **Consistency**: 100% deterministic identical outputs across repeated requests

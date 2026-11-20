import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.model_loader import ModelLoader
from api.preprocessing import clean_prediction_data
from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionResponse,
    WineInput,
)
from api.monitoring_logger import log_inference

model_loader = ModelLoader()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[INFO] Starting Wine Quality Prediction API...")
    try:
        model_loader.load()
    except Exception as exc:
        print(f"[WARNING] Could not preload model at startup: {exc}")
    yield
    print("[INFO] Wine Quality Prediction API shutting down...")


app = FastAPI(
    title="Wine Quality Prediction API",
    description="Lab 10 - ML Model Serving and API Engineering for Red Wine Quality",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = uuid.uuid4().hex[:12]
    errors = []

    for error in exc.errors():
        locations = [str(location) for location in error["loc"] if location != "body"]
        field = ".".join(locations)
        errors.append({
            "field": field,
            "message": error["msg"],
        })

    log_inference(
        input_data={},
        latency_ms=None,
        status="failure",
        error="Request validation failed",
        model_name=getattr(model_loader, "model_name", None),
        model_version=getattr(model_loader, "model_version", None),
        run_id=getattr(model_loader, "run_id", None),
        request_id=request_id,
    )

    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "message": "Request validation failed",
            "details": errors,
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": "Prediction service unavailable",
            "details": [
                {
                    "field": "prediction",
                    "message": str(exc.detail),
                }
            ],
        },
    )


@app.get("/")
def root():
    return {
        "message": "Wine Quality Prediction API is running",
        "lab": "Lab 10",
        "status": "success",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


@app.get("/metadata")
def metadata():
    return {
        "service": "Wine Quality Prediction API",
        "version": "1.0.0",
        "model_name": getattr(model_loader, "model_name", "Wine_Quality_Production_Model"),
        "model_uri": getattr(model_loader, "model_uri", "deployment/model.pkl"),
        "model_loaded": model_loader.model is not None,
        "preprocessor_loaded": model_loader.preprocessor is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(data: WineInput):
    request_id = uuid.uuid4().hex[:12]
    start_time = time.perf_counter()
    data_dict = data.model_dump(by_alias=True)

    if not model_loader.is_ready():
        latency_ms = round((time.perf_counter() - start_time) * 1000, 3)
        log_inference(
            input_data=data_dict,
            latency_ms=latency_ms,
            status="failure",
            error="Model or preprocessor is not available",
            model_name=model_loader.model_name,
            model_version=model_loader.model_version,
            run_id=model_loader.run_id,
            request_id=request_id,
        )
        raise HTTPException(
            status_code=503,
            detail="Model or preprocessor is not available",
        )

    try:
        df = pd.DataFrame([data_dict])
        df = clean_prediction_data(df)
        X = model_loader.preprocessor.transform(df)

        prediction = model_loader.model.predict(X)[0]
        probability = model_loader.model.predict_proba(X)[0][1]

        latency_ms = round((time.perf_counter() - start_time) * 1000, 3)

        log_inference(
            input_data=data_dict,
            prediction=int(prediction),
            quality_probability=float(probability),
            churn_probability=float(probability),
            latency_ms=latency_ms,
            status="success",
            model_name=model_loader.model_name,
            model_version=model_loader.model_version,
            run_id=model_loader.run_id,
            request_id=request_id,
        )

        return {
            "prediction": int(prediction),
            "quality_probability": float(probability),
            "churn_probability": float(probability),
        }

    except Exception as e:
        latency_ms = round((time.perf_counter() - start_time) * 1000, 3)
        log_inference(
            input_data=data_dict,
            latency_ms=latency_ms,
            status="failure",
            error=str(e),
            model_name=model_loader.model_name,
            model_version=model_loader.model_version,
            run_id=model_loader.run_id,
            request_id=request_id,
        )
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred during prediction: {e}",
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(data: BatchPredictionRequest):
    batch_request_id = uuid.uuid4().hex[:12]
    start_time = time.perf_counter()

    items = data.get_items()
    if not items:
        raise HTTPException(
            status_code=422,
            detail="Empty batch provided",
        )

    data_dicts = [item.model_dump(by_alias=True) for item in items]

    if not model_loader.is_ready():
        latency_ms = round((time.perf_counter() - start_time) * 1000, 3)
        for data_dict in data_dicts:
            log_inference(
                input_data=data_dict,
                latency_ms=latency_ms,
                status="failure",
                error="Model or preprocessor is not available",
                model_name=model_loader.model_name,
                model_version=model_loader.model_version,
                run_id=model_loader.run_id,
                request_id=batch_request_id,
            )
        raise HTTPException(
            status_code=503,
            detail="Model or preprocessor is not available",
        )

    try:
        df = pd.DataFrame(data_dicts)
        df = clean_prediction_data(df)
        X = model_loader.preprocessor.transform(df)

        predictions = model_loader.model.predict(X)
        probabilities = model_loader.model.predict_proba(X)[:, 1]

        total_latency_ms = round((time.perf_counter() - start_time) * 1000, 3)
        results = []

        for data_dict, pred, prob in zip(data_dicts, predictions, probabilities):
            log_inference(
                input_data=data_dict,
                prediction=int(pred),
                quality_probability=float(prob),
                churn_probability=float(prob),
                latency_ms=total_latency_ms,
                status="success",
                model_name=model_loader.model_name,
                model_version=model_loader.model_version,
                run_id=model_loader.run_id,
                request_id=uuid.uuid4().hex[:12],
            )
            results.append({
                "prediction": int(pred),
                "quality_probability": float(prob),
                "churn_probability": float(prob),
            })

        return {
            "count": len(results),
            "predictions": results,
        }

    except Exception as e:
        latency_ms = round((time.perf_counter() - start_time) * 1000, 3)
        for data_dict in data_dicts:
            log_inference(
                input_data=data_dict,
                latency_ms=latency_ms,
                status="failure",
                error=str(e),
                model_name=model_loader.model_name,
                model_version=model_loader.model_version,
                run_id=model_loader.run_id,
                request_id=uuid.uuid4().hex[:12],
            )
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred during batch prediction: {e}",
        )

"""
inference.py
============
Inference endpoint untuk model Diabetes (Kriteria 4).

- Load model dari MLflow (DagsHub run / local fallback)
- Serve via FastAPI di port 8000
- Metrik Prometheus diekspos di /metrics oleh prometheus_exporter.py (port 9091)

Usage:
    python inference.py
    # POST /predict {"Pregnancies": 6, "Glucose": 148, ...}
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import prometheus_client
from prometheus_client import (
    Counter, Gauge, Histogram, start_http_server,
)

# ---------------------------------------------------------------
# Metrik Prometheus (12) — di-expose di /metrics (port 8000)
# ---------------------------------------------------------------
REQUEST_TOTAL = Counter(
    "inference_request_total", "Total HTTP request", ["endpoint", "status"]
)
REQUEST_ERRORS = Counter("inference_request_errors_total", "Total request gagal (>=400)")
PREDICTION_POSITIVE = Counter("inference_prediction_positive_total", "Total prediksi diabetes (1)")
PREDICTION_NEGATIVE = Counter("inference_prediction_negative_total", "Total prediksi non-diabetes (0)")
PROBABILITY_MEAN = Gauge("inference_probability_mean", "Probability terakhir")
MODEL_VERSION = Gauge("model_version", "Model version (hash)")
UPTIME = Gauge("model_uptime_seconds", "Uptime server")
REQUESTS_IN_FLIGHT = Gauge("inference_requests_in_flight", "Request in flight")
LATENCY = Histogram(
    "inference_latency_seconds", "Latency predict",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
)
FEATURE_GLUCOSE_MEAN = Gauge("input_glucose_mean", "Rata-rata input Glucose (window)")
FEATURE_BMI_MEAN = Gauge("input_bmi_mean", "Rata-rata input BMI (window)")
PROCESS_RSS = Gauge("inference_process_memory_bytes", "Memory process (bytes)")

# ---------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------
MODEL_RUN_ID = os.environ.get("MODEL_RUN_ID", "a7f8f765378640ab954e5fcb2d2d4b88")
DAGSHUB_TOKEN = os.environ.get("DAGSHUB_TOKEN", "")
FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]

app = FastAPI(title="Diabetes Inference API", version="1.0")
_model = None
_start_time = time.time()


def load_model():
    """Load model dari DagsHub (online) atau local tracking."""
    global _model
    if _model is not None:
        return _model

    if DAGSHUB_TOKEN:
        import dagshub
        dagshub.init(repo_owner="sktamalik", repo_name="diabetes-mlflow", mlflow=True)
        uri = f"runs:/{MODEL_RUN_ID}/model"
        print(f"Load model dari DagsHub: {uri}")
        _model = mlflow.sklearn.load_model(uri)
    else:
        # Fallback: model sudah di-download ke local path (dl_model/model)
        local_path = Path(__file__).resolve().parent / "dl_model" / "model"
        if not local_path.exists():
            local_path = Path(__file__).resolve().parents[1] / "dl_model" / "model"
        print(f"Load model dari local path: {local_path}")
        _model = mlflow.sklearn.load_model(str(local_path))
    return _model


class DiabetesInput(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float


@app.on_event("startup")
async def startup():
    load_model()
    # Start exporter thread metrik (port 9091)
    start_http_server(9091)
    MODEL_VERSION.set(20260921)
    print("Exporter Prometheus di :9091/metrics")


@app.get("/health")
def health():
    return {"status": "ok", "uptime_seconds": round(time.time() - _start_time, 1)}


@app.middleware("http")
async def metrics_middleware(request, call_next):
    REQUESTS_IN_FLIGHT.inc()
    start = time.time()
    try:
        response = await call_next(request)
        status = response.status_code
    except Exception:
        status = 500
        raise
    finally:
        elapsed = time.time() - start
        REQUESTS_IN_FLIGHT.dec()
        REQUEST_TOTAL.labels(endpoint=request.url.path, status=str(status)).inc()
        if status >= 400:
            REQUEST_ERRORS.inc()
        UPTIME.set(time.time() - _start_time)
        try:
            PROCESS_RSS.set(__import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss * 1024)
        except Exception:
            PROCESS_RSS.set(0)
    return response


@app.post("/predict")
def predict(payload: DiabetesInput):
    start = time.time()
    model = load_model()
    x = np.array([[getattr(payload, f) for f in FEATURES]])
    df = pd.DataFrame(x, columns=FEATURES)

    pred = int(model.predict(df)[0])
    proba = float(model.predict_proba(df)[0][1])
    elapsed = time.time() - start

    LATENCY.observe(elapsed)
    if pred == 1:
        PREDICTION_POSITIVE.inc()
    else:
        PREDICTION_NEGATIVE.inc()
    PROBABILITY_MEAN.set(proba)
    FEATURE_GLUCOSE_MEAN.set(payload.Glucose)
    FEATURE_BMI_MEAN.set(payload.BMI)

    return {
        "prediction": pred,
        "probability_diabetes": round(proba, 4),
        "model_run_id": MODEL_RUN_ID,
        "latency_ms": round(elapsed * 1000, 2),
    }


# Mount /metrics default prometheus_client di API (port 8000)
from prometheus_client import make_asgi_app
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

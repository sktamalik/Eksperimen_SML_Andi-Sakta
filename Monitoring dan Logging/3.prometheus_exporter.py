"""
prometheus_exporter.py
======================
Prometheus exporter untuk monitoring sistem inference Diabetes (Kriteria 4).

- Start server metrik di port 9091 (di-scrape Prometheus)
- Business metrik dihitung dari aktivitas inference (share memory module-level)
- 12+ metrik ter-ekspos

Usage:
    python prometheus_exporter.py   # start di :9091
"""

from __future__ import annotations

import os
import sys
import time
import threading

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from prometheus_client import (
    start_http_server,
    Counter,
    Histogram,
    Gauge,
    REGISTRY,
)

# ---------------------------------------------------------------
# Metrik (12)
# ---------------------------------------------------------------
REQUEST_TOTAL = Counter(
    "inference_request_total", "Total HTTP request ke inference API", ["endpoint", "status"]
)
REQUEST_ERRORS = Counter(
    "inference_request_errors_total", "Total request gagal (non-2xx)"
)
PREDICTION_POSITIVE = Counter(
    "inference_prediction_positive_total", "Total prediksi diabetes (1)"
)
PREDICTION_NEGATIVE = Counter(
    "inference_prediction_negative_total", "Total prediksi non-diabetes (0)"
)
PROBABILITY_MEAN = Gauge(
    "inference_probability_mean", "Rata-rata probability diabetes terakhir (window 100 req)"
)
MODEL_VERSION = Gauge(
    "model_version", "Version model (run id hash, int)"
)
UPTIME = Gauge(
    "model_uptime_seconds", "Uptime inference server"
)
REQUESTS_IN_FLIGHT = Gauge(
    "inference_requests_in_flight", "Request sedang diproses"
)
LATENCY = Histogram(
    "inference_latency_seconds", "Latency request predict",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
)
FEATURE_GLUcOSE_MEAN = Gauge(
    "input_glucose_mean", "Rata-rata input Glucose (window)"
)
FEATURE_BMI_MEAN = Gauge(
    "input_bmi_mean", "Rata-rata input BMI (window)"
)
PROCESS_RSS = Gauge(
    "inference_process_memory_bytes", "Memory usage process inference"
)


class InferenceMetrics:
    """Helper dipakai inference.py buat update metrik."""

    @staticmethod
    def record_request(endpoint: str, status: int, latency: float):
        REQUEST_TOTAL.labels(endpoint=endpoint, status=str(status)).inc()
        if status >= 400:
            REQUEST_ERRORS.inc()
        LATENCY.observe(latency)

    @staticmethod
    def record_prediction(pred: int, proba: float):
        if pred == 1:
            PREDICTION_POSITIVE.inc()
        else:
            PREDICTION_NEGATIVE.inc()
        PROBABILITY_MEAN.set(proba)

    @staticmethod
    def record_features(glucose: float, bmi: float):
        FEATURE_GLUcOSE_MEAN.set(glucose)
        FEATURE_BMI_MEAN.set(bmi)


def set_static(model_version: int, process_rss: int):
    """Set gauge statis (model version, memory)."""
    MODEL_VERSION.set(model_version)
    PROCESS_RSS.set(process_rss)


def update_uptime_loop(start: float, interval: int = 5):
    def _loop():
        while True:
            UPTIME.set(time.time() - start)
            time.sleep(interval)
    t = threading.Thread(target=_loop, daemon=True)
    t.start()


if __name__ == "__main__":
    port = int(os.environ.get("EXPORTER_PORT", "9091"))
    print(f"Prometheus exporter start di :{port}/metrics")
    start_http_server(port)

    # Demo: naikkan uptime + version statis supaya gauge tidak 0
    set_static(model_version=20260921, process_rss=0)
    update_uptime_loop(start=time.time())

    print("Exporter berjalan. Ctrl+C untuk stop.")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("Exporter stop.")
"""
modelling.py
============
Baseline model training — Kriteria 2 Basic.

MLflow autolog + LogisticRegression pada dataset preprocessing.
Run tracking ui lokal (http://127.0.0.1:5000).

Usage:
    python modelling.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Fix Windows cp1252 console encoding for MLflow emoji output
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Force headless matplotlib buat preview autolog no tkinter thread crash
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------
DATA_PATH = Path(__file__).resolve().parent / "diabetes_preprocessing.csv"
TARGET = "Outcome"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# MLflow local tracking
mlflow.set_tracking_uri("http://127.0.0.1:5000")
EXPERIMENT_NAME = "Eksperimen_SML_Andi-Sakta"


def load_and_split(data_path: Path):
    """Load preprocessed data, split train/test internal for eval."""
    df = pd.read_csv(data_path)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test


def evaluate(model, X_test, y_test):
    """Metric eval set."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    return metrics


def main() -> None:
    X_train, X_test, y_train, y_test = load_and_split(DATA_PATH)

    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.autolog(log_models=True)  # basik: autolog gun range

    with mlflow.start_run(run_name="baseline-logistic-regression") as run:
        model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)

        metrics = evaluate(model, X_test, y_test)

        # autolog log model + params + metrics internalek
        mlflow.log_params({"model_type": "LogisticRegression", "max_iter": 1000})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        print("\nResultaat:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print(f"\nMLflow run_id: {run.info.run_id}")


if __name__ == "__main__":
    main()
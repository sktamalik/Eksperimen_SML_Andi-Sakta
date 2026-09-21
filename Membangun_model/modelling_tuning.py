"""
modelling_tuning.py
===================
Hyperparameter tuning + manual logging — Kriteria 2 Skilled/Advance.

- GridSearchCV RandomForest
- Manual logging (no autolog) dengan metrik: accuracy, precision, recall, f1, roc_auc
- Artefak tambahan: confusion_matrix.png, roc_curve.png, feature_importance.png
- MLflow tracking online ke DagsHub (advance)

Usage:
    set DAGSHUB_TOKEN=<token> terlebih dahulu
    python modelling_tuning.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split

DATA_PATH = Path(__file__).resolve().parent / "diabetes_preprocessing.csv"
TARGET = "Outcome"
TEST_SIZE = 0.2
RANDOM_STATE = 42

DAGSHUB_REPO = "sktamalik/diabetes-mlflow"
DAGSHUB_TOKEN = os.environ.get("DAGSHUB_TOKEN", "")


def setup_tracking() -> None:
    """Setup MLflow online DagsHub; fallback local kalau tak ada token."""
    if DAGSHUB_TOKEN:
        import dagshub
        dagshub.init(repo_owner="sktamalik", repo_name="diabetes-mlflow", mlflow=True)
        # dagshub.init otomatis set tracking uri ke https://dagshub.com/<owner>/<repo>.mlflow
        print(f"Tracking online: DagsHub {DAGSHUB_REPO}")
    else:
        # Fallback local (development di Windows)
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        print("⚠️  DAGSHUB_TOKEN kosong — pakai MLflow local 127.0.0.1:5000.")


def load_data() -> tuple:
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


def log_confusion_matrix(model, X_test, y_test, run) -> None:
    """Artefak 1: confusion matrix."""
    cm = confusion_matrix(y_test, model.predict(X_test))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Negatif", "Positif"]); ax.set_yticklabels(["Negatif", "Positif"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual")
    ax.set_title("Confusion Matrix — RandomForest Tuned")
    fig.colorbar(im)
    plt.tight_layout()
    path = Path(__file__).parent / "confusion_matrix.png"
    fig.savefig(path)
    plt.close(fig)
    mlflow.log_artifact(str(path), artifact_path="artifacts")
    print(f"  [artefak] confusion_matrix.png: {cm.tolist()}")


def log_roc_curve(model, X_test, y_test, run) -> None:
    """Artefak 2: ROC curve."""
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})", lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — RandomForest Tuned")
    ax.legend()
    plt.tight_layout()
    path = Path(__file__).parent / "roc_curve.png"
    fig.savefig(path)
    plt.close(fig)
    mlflow.log_artifact(str(path), artifact_path="artifacts")
    print(f"  [artefak] roc_curve.png (AUC={auc:.4f})")


def log_feature_importance(model, X_train, run) -> None:
    """Artefak 3: feature importance."""
    imp = model.feature_importances_
    feat_names = X_train.columns
    fig, ax = plt.subplots(figsize=(8, 5))
    idx = np.argsort(imp)[::-1]
    ax.barh([feat_names[i] for i in idx][::-1], imp[idx][::-1])
    ax.set_xlabel("Importance"); ax.set_title("Feature Importance — RandomForest Tuned")
    plt.tight_layout()
    path = Path(__file__).parent / "feature_importance.png"
    fig.savefig(path)
    plt.close(fig)
    mlflow.log_artifact(str(path), artifact_path="artifacts")
    print(f"  [artefak] feature_importance.png: {dict(zip(feat_names[idx][:3], imp[idx][:3].round(3)))}")


def main() -> None:
    X_train, X_test, y_train, y_test = load_data()

    setup_tracking()
    mlflow.set_experiment("Eksperimen_SML_Andi-Sakta-tuning")

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    }
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    gs = GridSearchCV(
        rf, param_grid, cv=5, scoring="roc_auc", n_jobs=-1, refit=True, verbose=0
    )

    with mlflow.start_run(run_name="tuning-randomforest-gridsearch") as run:
        # --- Manual logging (skilled kriteria) ---
        gs.fit(X_train, y_train)
        best = gs.best_estimator_

        # Log hyperparameter
        mlflow.log_params({f"grid_{k}": v for k, v in gs.best_params_.items()})
        mlflow.log_params({"cv_k": 5, "scoring": "roc_auc", "model_type": "RandomForest", "tuning": "GridSearchCV"})

        # Eval fold
        y_pred = best.predict(X_test)
        y_proba = best.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }
        mlflow.log_metrics(metrics)
        mlflow.log_metric("cv_best_mean_test_score", gs.best_score_)

        # Artefak
        log_confusion_matrix(best, X_test, y_test, run)
        log_roc_curve(best, X_test, y_test, run)
        log_feature_importance(best, X_train, run)

        # Logging model (manual)
        mlflow.sklearn.log_model(best, "model", input_example=X_test.iloc[:1].to_dict("records"))

        print(f"\nBest params: {gs.best_params_}")
        print(f"CV best roc_auc: {gs.best_score_:.4f}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print(f"\nMLflow run_id: {run.info.run_id}")
        print(f"Tracked at: {mlflow.get_tracking_uri()}")


if __name__ == "__main__":
    main()
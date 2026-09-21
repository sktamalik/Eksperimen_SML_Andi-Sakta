# Plan Submission SMSML — Andi Sakta

Target: **Advance (4)** di semua 4 kriteria.
Dataset: **Pima Indians Diabetes** (classification biner, target `Outcome`).
Stack: Python 3.12.7 (pyenv), SKLearn, MLflow 2.19. CPU-only.
Akun: GitHub `sktamalik`, Docker Hub `sktamalik`, DagsHub `sktamalik/diabetes-mlflow`.

## Struktur repo

- **Repo ini** (`Eksperimen_SML_Andi-Sakta`, remote diubah) → kriteria 1 + 2
- **Repo terpisah** (`Workflow-CI`) → kriteria 3
- Output: `SMSML_Andi-Sakta.zip` (gabung, no nested ZIP) + 2 link repo public

## Kriteria 1 — Eksperimen + automasi (advance)

`Eksperimen_SML_Andi-Sakta/`
- `diabetes_raw.csv` — dataset (di-commit ke repo, tak pakai Kaggle di CI)
- `preprocessing/Eksperimen_Andi-Sakta.ipynb` — template MSML: loading, EDA, preprocessing
- `preprocessing/automate_Andi-Sakta.py` — konversi notebook → fungsi, output siap-train
- `preprocessing/diabetes_preprocessing.csv` — hasil
- `.github/workflows/preprocessing.yml` — Actions jalankan automate.py, commit hasil terbaru

Preprocessing: imputasi 0→NaN→median (Glucose, BloodPressure, SkinThickness, Insulin, BMI), split 80/20 stratify, StandardScaler fit-on-train-only.

## Kriteria 2 — Membangun_model (advance)

`Membangun_model/`
- `modelling.py` — MLflow autolog, LogisticRegression baseline
- `modelling_tuning.py` — manual logging, GridSearchCV RandomForest, tuning
- `DagsHub.txt` — link DagsHub
- `screenshoot_dashboard.jpg`, `screenshoot_artifak.jpg`, `requirements.txt`

Advance: MLflow online di DagsHub (`dagshub.init`), manual logging + **3 artefak tambahan**: confusion_matrix.png, roc_curve.png, feature_importance.png.
Metrik: accuracy, precision, recall, f1, roc_auc.

## Kriteria 3 — Workflow-CI (advance)

`Workflow-CI/`
- `MLProject/modelling.py`, `conda.yaml`, `MLProject`, `diabetes_preprocessing.csv`
- `.github/workflows/ci.yml` — retrain saat trigger
- Docker: `mlflow build-docker` → push `sktamalik/diabetes-ml:latest` ke Docker Hub

Secrets: `DAGSHUB_TOKEN`, `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`.

## Kriteria 4 — Monitoring & Logging (advance)

- `1.bukti_serving` — pull image / mlflow serve
- `2.prometheus.yml`, `3.prometheus_exporter.py`, `7.inference.py`
- Grafana: **12 metrik** + **3 alerting**, nama dashboard = username Dicoding `sktamalik`

Metrik: http_request_total, latency histogram (p50/p95/p99), errors_total, prediction_positive/negative_total, probability_mean, model_version, uptime, input feature means, requests_in_flight, memory/cpu.
Alerting: latency p95>2s, error rate>5%, positive-rate abnormal.

## Env setup (sebelum implementasi)

1. Set remote folder ini → `https://github.com/sktamalik/Eksperimen_SML_Andi-Sakta.git`
2. Setup `~/.kaggle/kaggle.json` (key dari user, disimpan file, tak di-chat)
3. pyenv/conda → Python 3.12.7
4. Install `gh` CLI
5. DagsHub setup akun + token; Docker Hub access token
6. Setup repo `Workflow-CI` + remote

## Urutan kerja

1. Env setup + dataset
2. Kriteria 1: notebook + automate.py + workflow
3. Kriteria 2: modelling.py + tuning + DagsHub
4. Kriteria 3: MLProject + CI + Docker
5. Kriteria 4: exporter + prometheus + grafana + inference
6. Screenshot semua bukti
7. Zip + submit

## Keamanan

- Kaggle key user sudah di-chat → anjurkan rotate di kaggle.com/settings
- Token/secret simpan di env/GitHub secret, tak di file repo

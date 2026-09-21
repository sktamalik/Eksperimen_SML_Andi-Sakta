# Serve MLflow model — Diabetes RandomForest
FROM python:3.12-slim

RUN apt-get -y update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir mlflow==2.19.0 scikit-learn pandas numpy

# Copy model
COPY model_dir/model /opt/ml/model

EXPOSE 8080

# mlflow serve local env, no env-manager (deps sudah di image)
ENTRYPOINT ["python", "-m", "mlflow", "models", "serve", "-m", "/opt/ml/model", "-h", "0.0.0.0", "-p", "8080", "--env-manager", "local"]
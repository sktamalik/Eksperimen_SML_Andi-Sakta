"""Traffic generator untuk bukti monitoring Grafana (Kriteria 4).

Kirim request ke /predict secara kontinu supaya panel berbasis rate()
mempunyai data pada window 1m/5m saat screenshot diambil.
"""
from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request

URL = "http://127.0.0.1:8000/predict"
random.seed(42)

BASE = {
    "Pregnancies": 0.65,
    "Glucose": 0.97,
    "BloodPressure": -0.82,
    "SkinThickness": 0.22,
    "Insulin": -0.22,
    "BMI": 0.44,
    "DiabetesPedigreeFunction": 0.65,
    "Age": -0.45,
}

NEG = [(-1.2, -2.0), (-1.0, -1.5), (-1.2, -1.0), (-0.8, -1.8), (-1.1, -2.2)]
POS = [(1.5, 1.2), (2.0, 1.8), (1.2, 1.0), (1.8, 2.2), (2.4, 1.5)]


def post(payload: dict) -> None:
    req = urllib.request.Request(
        URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        urllib.request.urlopen(req, timeout=5).read()
    except Exception:
        pass


def main() -> None:
    i = 0
    while True:
        # mayoritas sukses, sesekali payload invalid supaya panel error rate hidup
        if i % 25 == 24:
            post({"Glucose": 1.0})
        else:
            base = dict(BASE)
            if i % 3 == 0:
                g, b = random.choice(NEG)
            else:
                g, b = random.choice(POS)
            base["Glucose"] = g
            base["BMI"] = b
            base["Age"] = round(random.uniform(-1.0, 2.0), 3)
            post(base)
        i += 1
        time.sleep(0.5)


if __name__ == "__main__":
    main()

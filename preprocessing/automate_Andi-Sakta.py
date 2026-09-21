"""
automate_Andi-Sakta.py
======================
ETL preprocessing otomatis — konversi tahapan notebook Eksperimen_Andi-Sakta.ipynb
menjadi script yang jalan tanpa intervensi manual.

Pipeline (identik dengan notebook):
  1. Load raw dataset (diabetes_raw.csv)
  2. EDA check konsol (shape, missing, duplikasi)
  3. Imputasi nilai 0 -> NaN -> median (Glucose, BloodPressure, SkinThickness, Insulin, BMI)
  4. Split train/test 80/20 stratified
  5. StandardScaler (fit on train only)
  6. Save hasil preprocessing -> diabetes_preprocessing.csv

Usage:
    python automate_Andi-Sakta.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------
# Konfigurasi tetap (mirror notebook eksperimen)
# ---------------------------------------------------------------
RAW_PATH = Path(__file__).resolve().parent.parent / "diabetes_raw.csv"
OUTPUT_PATH = Path(__file__).resolve().parent / "diabetes_preprocessing.csv"

COLS_WITH_ZERO = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
TARGET = "Outcome"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data(path: Path) -> pd.DataFrame:
    """Load raw dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset {path} tidak ditemukan.")
    df = pd.read_csv(path)
    return df


def eda_report(df: pd.DataFrame) -> None:
    """Print EDA ringkas konsol (mirror notebook)."""
    print("=" * 60)
    print("EDA REPORT")
    print("=" * 60)
    print(f"Shape: {df.shape}")
    print(f"\nMissing:\n{df.isnull().sum()}")
    print(f"\nDuplicated: {df.duplicated().sum()}")
    print(f"\nOutcome value counts:\n{df['Outcome'].value_counts(normalize=True)}")
    for c in COLS_WITH_ZERO:
        zero_count = (df[c] == 0).sum()
        if zero_count:
            print(f"  [{c}] nilai 0: {zero_count}")


def impute_zero_with_median(df: pd.DataFrame) -> pd.DataFrame:
    """Nilai 0 (tidak valid) -> NaN -> median kolom."""
    df_clean = df.copy()
    for c in COLS_WITH_ZERO:
        df_clean[c] = df_clean[c].replace(0, np.nan)
        df_clean[c] = df_clean[c].fillna(df_clean[c].median())
    return df_clean


def split_scale(df_clean: pd.DataFrame):
    """Split 80/20 stratified + StandardScaler fit-on-train."""
    X = df_clean.drop(columns=[TARGET])
    y = df_clean[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X, X_train_scaled, X_test_scaled, y_train, y_test, scaler


def build_output(X_train_scaled, X_test_scaled, y_train, y_test, columns) -> pd.DataFrame:
    """Gabung train/test kembali dengan kolom asli."""
    train_df = pd.DataFrame(X_train_scaled, columns=columns)
    train_df[TARGET] = y_train.values
    test_df = pd.DataFrame(X_test_scaled, columns=columns)
    test_df[TARGET] = y_test.values
    return pd.concat([train_df, test_df], axis=0)


def main() -> None:
    """Pipeline utama."""
    print(f"[1/6] Load raw data: {RAW_PATH}")
    df = load_data(RAW_PATH)

    print("[2/6] EDA report...")
    eda_report(df)

    print("[3/6] Imputasi 0 -> median...")
    df_clean = impute_zero_with_median(df)
    assert df_clean.isnull().sum().sum() == 0, "Missing tidak ful belum."
    print(f"       Missing setelah imputasi: {df_clean.isnull().sum().sum()}")

    print("[4/6] Split 80/20 stratified + scaling...")
    X, X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_scale(df_clean)

    print(f"       X_train: {X_train_scaled.shape}, X_test: {X_test_scaled.shape}")
    print(f"       Scaler fit on train only. Mean train: {X_train_scaled.mean(axis=0).round(3)}")

    print("[5/6] Build output DataFrame...")
    df_preprocessed = build_output(X_train_scaled, X_test_scaled, y_train, y_test, X.columns)

    print(f"[6/6] Save output: {OUTPUT_PATH}")
    df_preprocessed.to_csv(OUTPUT_PATH, index=False)
    print(f"       Done. Shape: {df_preprocessed.shape}, clean: {df_preprocessed.isnull().sum().sum()}")


if __name__ == "__main__":
    main()
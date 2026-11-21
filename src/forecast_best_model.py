"""Level 2 input: write the 16-day forecast with the BEST trained model.

Run from the project root (after python src/run_pipeline.py):
    python src/forecast_best_model.py

It writes the same files as src/forecast_naive.py, so src/inventory.py and the
dashboard work with either one without any change:

    reports/forecast_16d.parquet     date, store_nbr, family, forecast, model
                                     (2017-08-16 .. 2017-08-31, the Kaggle test days)
    reports/val_predictions.parquet  date, store_nbr, family, sales, prediction
                                     (the same model on validation, July 2017 -> sigma)
    reports/recent_sales.parquet     date, store_nbr, family, sales (last 60 real days)

The model is the one train_models.py saved as the best (lowest validation WAPE):
models/best_model.pkl. It was trained on data up to TRAIN_END (2017-06-30).
Its clues for 16-31 Aug come from data/processed/test_features.parquet, built by
feature_engineering.py; every sales clue there is at least 16 days old, so the
forecast only uses sales up to 2017-08-15 ("today").
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd

import config
from data_io import setup_console

KEYS = ["store_nbr", "family"]
HORIZON = 16
RECENT_DAYS = 60

RESULTS_PATH = config.REPORTS_DIR / "model_comparison_results.parquet"
FORECAST_PATH = config.REPORTS_DIR / "forecast_16d.parquet"
VAL_PREDICTIONS_PATH = config.REPORTS_DIR / "val_predictions.parquet"
RECENT_SALES_PATH = config.REPORTS_DIR / "recent_sales.parquet"


def main() -> int:
    setup_console()
    for path in (config.BEST_MODEL_PATH, config.FEATURES_TEST_PATH, RESULTS_PATH):
        if not path.exists():
            print(f"{path.name} not found. Run first:  python src/run_pipeline.py")
            return 1

    pack = joblib.load(config.BEST_MODEL_PATH)
    name, features = pack["name"], pack["features"]
    transform = pack.get("target_transform", "none")
    print(f"Best model: {name}  ({len(features)} clues, target transform: {transform})")

    # 1. The next 16 days
    columns = list(dict.fromkeys(["date"] + KEYS + features))   # store_nbr is a key AND a clue
    future = pd.read_parquet(config.FEATURES_TEST_PATH, columns=columns)
    pred = pack["model"].predict(future[features].to_numpy(dtype="float32"))
    if transform == "log1p":
        pred = np.expm1(pred)
    forecast = future[["date"] + KEYS].copy()
    forecast["forecast"] = np.clip(pred, 0, None)
    forecast["model"] = name
    if forecast["date"].nunique() != HORIZON:
        print(f"ERROR: expected {HORIZON} forecast days, found {forecast['date'].nunique()}")
        return 1

    # 2. The same model on validation (its errors give sigma for the safety stock)
    results = pd.read_parquet(RESULTS_PATH)
    val = results[results["model"] == name].rename(
        columns={"actual_sales": "sales", "predicted_sales": "prediction"})
    val = val[["date"] + KEYS + ["sales", "prediction"]]
    val["prediction"] = val["prediction"].clip(lower=0)

    # 3. Recent real sales, for the charts
    history = pd.read_parquet(config.TRAIN_CLEAN_PATH, columns=["date"] + KEYS + ["sales"])
    day0 = history["date"].max()
    recent = history[history["date"] > day0 - pd.Timedelta(days=RECENT_DAYS)]

    forecast.to_parquet(FORECAST_PATH, index=False)
    val.to_parquet(VAL_PREDICTIONS_PATH, index=False)
    recent.to_parquet(RECENT_SALES_PATH, index=False)

    print(f"Day 0 (today): {day0.date()}")
    print(f"Forecast: {forecast['date'].min().date()} -> {forecast['date'].max().date()}, "
          f"{len(forecast):,} rows -> {FORECAST_PATH.relative_to(config.PROJECT_ROOT)}")
    print(f"Validation predictions: {len(val):,} rows -> "
          f"{VAL_PREDICTIONS_PATH.relative_to(config.PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

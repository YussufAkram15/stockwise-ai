"""STEPS 7-8 (Zenab): Polynomial Regression (degree 2) on a FEW numeric clues.

Run from the project root (after prepare_data.py and feature_engineering.py):
    python src/train_polynomial.py
    python src/train_polynomial.py --no-mlflow      # skip MLflow logging

Reads  : data/processed/train_features.parquet   (Adham's feature table)
Writes : reports/polynomial_results.json            scores + settings (copy them into the PR)
         reports/polynomial_val_predictions.parquet validation predictions, same columns as
                                                    Sara's model_comparison_results.parquet
         mlflow.db                                  MLflow run (NOT committed to GitHub)

Why only 6 clues?
    Degree 2 multiplies every clue with every other clue (and with itself).
    6 clues -> 6 + 21 = 27 columns.   38 clues -> ~780 columns x 2.9M rows = out of memory.

Pipeline (the order matters):
    SimpleImputer(median) -> StandardScaler -> PolynomialFeatures(2) -> LinearRegression
    Scaling BEFORE the polynomial step, because squaring big numbers (sales) makes huge values.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import config
from data_io import setup_console

# ----------------------------------------------------------------------------
# Settings (agreed with the team)
# ----------------------------------------------------------------------------
DEGREE = 2
FEATURES = [
    "sales_lag_16", "sales_lag_21", "sales_lag_28",     # what sold 16 / 21 / 28 days ago
    "sales_roll_7_mean", "sales_roll_28_mean",          # weekly / monthly average (already shifted)
    "onpromotion",                                      # promotions today
]
# NOT used on purpose: family_enc, city_enc, state_enc, store_nbr ... they are labels, not amounts.

# EXPERIMENT (--more-clues): 3 extra clues chosen from the EDA, not from the validation score:
#   is_weekend (weekends sell +39%), is_holiday (national holidays +15%), store_family_mean_sales (size of the series).
# (is_payday is left out: the EDA found no payday effect.)
EXTRA_FEATURES = ["is_weekend", "is_holiday", "store_family_mean_sales"]

TARGET = "sales"
ID_COLS = ["id", "date", "store_nbr", "family"]
QC_COLS = ["qc_zero_series", "qc_before_first_sale"]
SUSPECT_COL = "qc_sales_suspect"      # extreme AND unexplained by holiday/event/promo: removed from training
OUTLIER_COL = "qc_sales_outlier"    # only used with --exclude-outliers (an experiment, not the agreed default)

# Split: the agreed dates, read from src/config.py (one place for the whole team).
TRAIN_END = config.TRAIN_END
VAL_START = config.VAL_START
VAL_END = config.VAL_END
TEST_START = config.TEST_START     # 31 Jul -> 15 Aug is the TEST slice: never touched here

NAIVE_BASELINE_WAPE = 17.50        # Yussuf's naive baseline on validation (the number to beat)

MLFLOW_URI = f"sqlite:///{(config.PROJECT_ROOT / 'mlflow.db').as_posix()}"   # = sqlite:///mlflow.db at the root
EXPERIMENT = "stockwise-ai"
RUN_NAME = f"polynomial_deg{DEGREE}"
MODEL_NAME = "PolynomialRegression"

FEATURES_PATH = getattr(config, "FEATURES_TRAIN_PATH", config.PROCESSED_DIR / "train_features.parquet")


# ----------------------------------------------------------------------------
# Scores: MAE, RMSE, WAPE  (TEMPORARY copy - switch to Sara's functions when she pushes them)
# ----------------------------------------------------------------------------
def mae(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - pred)))


def rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def wape(actual: np.ndarray, pred: np.ndarray) -> float:
    """100 * sum|actual - pred| / sum(actual)   (in %)."""
    total = float(np.sum(actual))
    return float("nan") if total <= 0 else float(100.0 * np.sum(np.abs(actual - pred)) / total)


def score_all(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    pred = np.clip(pred, 0, None)          # sales cannot be negative
    return {"val_mae": mae(actual, pred), "val_rmse": rmse(actual, pred), "val_wape": wape(actual, pred)}


# ----------------------------------------------------------------------------
# Data + split
# ----------------------------------------------------------------------------
def load_and_split(features: list[str], exclude_outliers: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not FEATURES_PATH.exists():
        raise SystemExit(f"{FEATURES_PATH} not found. Run first:  python src/feature_engineering.py")

    import pyarrow.parquet as pq
    available = set(pq.read_schema(FEATURES_PATH).names)
    use_suspect = SUSPECT_COL in available            # present after the cleaning step with suspected data errors
    needed = (ID_COLS + [TARGET] + QC_COLS + features
              + ([SUSPECT_COL] if use_suspect else []) + ([OUTLIER_COL] if exclude_outliers else []))
    missing = [c for c in needed if c not in available]
    if missing:
        raise SystemExit(f"Columns missing from {FEATURES_PATH.name}: {missing}")

    df = pd.read_parquet(FEATURES_PATH, columns=needed)
    df["date"] = pd.to_datetime(df["date"])

    # Train: up to TRAIN_END, WITHOUT the rows flagged by the cleaning step.
    qc_ok = (df["qc_zero_series"].fillna(0) == 0) & (df["qc_before_first_sale"].fillna(0) == 0)
    train_mask = (df["date"] <= pd.Timestamp(TRAIN_END)) & qc_ok
    if use_suspect:
        # Suspected DATA ERRORS (extreme AND unexplained). Real peaks are kept. Training rows only.
        is_suspect = df[SUSPECT_COL].fillna(0) == 1
        print(f"[POLY] Suspected data-error rows removed from TRAIN only: {int((train_mask & is_suspect).sum()):,}")
        train_mask = train_mask & ~is_suspect
    if exclude_outliers:
        # EXPERIMENT: also drop the unusually high sales from TRAINING only (validation is never touched).
        is_outlier = df[OUTLIER_COL].fillna(0) == 1
        print(f"[POLY] Outlier rows removed from TRAIN only: {int((train_mask & is_outlier).sum()):,}")
        train_mask = train_mask & ~is_outlier
    train = df[train_mask]
    # Validation: all rows of July 2017 (no qc filter, we score the real thing).
    val = df[(df["date"] >= pd.Timestamp(VAL_START)) & (df["date"] <= pd.Timestamp(VAL_END))]

    if train.empty or val.empty:
        raise SystemExit(f"Empty split: train rows={len(train):,}, validation rows={len(val):,}")
    if val["date"].max() >= pd.Timestamp(TEST_START):
        raise SystemExit("Validation overlaps the test slice - stop.")
    if train[TARGET].isna().any() or val[TARGET].isna().any():
        raise SystemExit("Missing sales in train/validation rows - check the feature table.")
    return train.reset_index(drop=True), val.reset_index(drop=True)


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),            # NaN in early rows -> median
        ("scaler", StandardScaler()),                             # BEFORE squaring
        ("poly", PolynomialFeatures(degree=DEGREE, include_bias=False)),
        ("model", LinearRegression(copy_X=False)),                # the polynomial table is not needed afterwards: saves RAM
    ])


# ----------------------------------------------------------------------------
def log_to_mlflow(params: dict, metrics: dict, run_name: str) -> bool:
    try:
        import mlflow
    except ImportError:
        print("mlflow is missing. Run:  pip install mlflow   then run again (or use --no-mlflow).")
        return False
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
    print(f"MLflow: run '{run_name}' logged in experiment '{EXPERIMENT}'  ({MLFLOW_URI})")
    return True


def main(argv: list[str] | None = None) -> int:
    setup_console()
    ap = argparse.ArgumentParser(description="StockWise AI: Polynomial Regression (degree 2).")
    ap.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging")
    ap.add_argument("--exclude-outliers", action="store_true",
                    help="EXPERIMENT: also remove qc_sales_outlier rows from the TRAINING rows")
    ap.add_argument("--more-clues", action="store_true",
                    help="EXPERIMENT: add is_weekend, is_holiday, store_family_mean_sales to the 6 clues")
    ap.add_argument("--log-target", action="store_true",
                    help="EXPERIMENT: learn log(1 + sales) instead of sales, then convert back")
    args = ap.parse_args(argv)
    # each experiment gets its own name, so the agreed run is never overwritten
    suffix = (("_more_clues" if args.more_clues else "") + ("_log" if args.log_target else "")
              + ("_no_outliers" if args.exclude_outliers else ""))
    features = FEATURES + EXTRA_FEATURES if args.more_clues else FEATURES
    run_name = RUN_NAME + suffix
    results_json = config.REPORTS_DIR / f"polynomial_results{suffix}.json"
    predictions_path = config.REPORTS_DIR / f"polynomial_val_predictions{suffix}.parquet"

    try:
        import pyarrow  # noqa: F401
    except ImportError:
        print("pyarrow is missing. Run:  pip install pyarrow")
        return 1

    print(f"[POLY] Loading {FEATURES_PATH.name} ...")
    train, val = load_and_split(features, args.exclude_outliers)
    print(f"[POLY] Train: {len(train):,} rows (<= {TRAIN_END}, qc rows removed)")
    print(f"[POLY] Val  : {len(val):,} rows ({VAL_START} -> {VAL_END})")
    print(f"[POLY] Clues ({len(features)}): {', '.join(features)}")

    X_train = train[features].to_numpy(dtype="float64")
    y_train = train[TARGET].to_numpy(dtype="float64")
    X_val = val[features].to_numpy(dtype="float64")
    y_val = val[TARGET].to_numpy(dtype="float64")

    t0 = time.time()
    pipe = build_pipeline()
    if args.log_target:
        # learn log(1 + sales); the largest allowed prediction is the largest sale seen in training
        pipe.fit(X_train, np.log1p(y_train))
        pred_raw = np.expm1(np.clip(pipe.predict(X_val), 0, np.log1p(y_train.max())))
    else:
        pipe.fit(X_train, y_train)
        pred_raw = pipe.predict(X_val)
    elapsed = time.time() - t0
    n_poly_cols = pipe.named_steps["poly"].n_output_features_

    metrics = score_all(y_val, pred_raw)
    clipped = float(np.mean(pred_raw < 0))
    print(f"\n[POLY] Polynomial columns after degree {DEGREE}: {n_poly_cols}   (fit + predict: {elapsed:.1f}s)")
    print(f"[POLY] Predictions below 0 (clipped to 0): {clipped:.2%}   largest prediction: {pred_raw.max():,.0f}")
    print(f"[POLY] Validation  MAE = {metrics['val_mae']:.3f}   RMSE = {metrics['val_rmse']:.3f}   "
          f"WAPE = {metrics['val_wape']:.2f}%")
    verdict = "BEATS" if metrics["val_wape"] < NAIVE_BASELINE_WAPE else "does NOT beat"
    print(f"[POLY] Naive baseline WAPE = {NAIVE_BASELINE_WAPE:.2f}%  ->  Polynomial {verdict} the baseline.")

    params = {
        "model": MODEL_NAME + suffix,
        "degree": DEGREE,
        "features": ", ".join(features),
        "n_features": len(features),
        "n_poly_columns": n_poly_cols,
        "pipeline": "SimpleImputer(median) > StandardScaler > PolynomialFeatures > LinearRegression",
        "train_end": TRAIN_END,
        "val_start": VAL_START,
        "val_end": VAL_END,
        "qc_filter_train": "qc_zero_series=0 and qc_before_first_sale=0 and qc_sales_suspect=0" + (" and qc_sales_outlier=0" if args.exclude_outliers else ""),
        "exclude_outliers": args.exclude_outliers,
        "more_clues": args.more_clues,
        "log_target": args.log_target,
        "n_train_rows": len(train),
        "n_val_rows": len(val),
    }

    # Save results BEFORE MLflow, so nothing is lost if logging fails.
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results_json.write_text(json.dumps({"params": params, "metrics": metrics,
                                        "naive_baseline_wape": NAIVE_BASELINE_WAPE}, indent=2), encoding="utf-8")
    out = val[["id", "date", "store_nbr", "family"]].copy()
    out["actual_sales"] = y_val
    out["model"] = MODEL_NAME + suffix
    out["predicted_sales"] = np.clip(pred_raw, 0, None)
    out.to_parquet(predictions_path, index=False)
    print(f"[POLY] Saved: {results_json}")
    print(f"[POLY] Saved: {predictions_path}")

    if args.no_mlflow:
        return 0
    try:
        return 0 if log_to_mlflow(params, metrics, run_name) else 1
    except Exception as exc:  # report it clearly, results are already saved
        print(f"MLflow logging FAILED: {exc!r}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
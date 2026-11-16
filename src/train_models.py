
"""STEP 8: Model training and artifact generation.

Run from the project root (after feature_engineering.py):
    python src/train_models.py
    python src/train_models.py --models lr rf
    python src/train_models.py --models xgb
    python src/train_models.py --no-mlflow

Split (time-based, never shuffle):
    Train : date <= TRAIN_END
    Val   : VAL_START -> VAL_END

Models trained:
    lr   LinearRegression
    rf   RandomForestRegressor
    xgb  XGBRegressor

Metric:
    RMSLE (Root Mean Squared Log Error).

Outputs:
    reports/model_comparison_results.parquet
    reports/model_comparison_metadata.json
    reports/best_model_predictions.parquet
    models/best_model.pkl

This script does NOT generate charts or Markdown reports.
Run compare_models.py separately to generate comparison reports and figures.

MLflow tracking is stored in mlruns/ when enabled.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
from data_io import setup_console

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Feature columns used by all models
# (Never include sales, qc_ flags, raw text, or the date itself)
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    # Calendar
    "day_of_week", "day_of_month", "week_of_year", "month", "quarter", "year",
    "is_weekend", "days_to_month_end", "is_payday",

    # Store / series encodings
    "store_nbr", "store_type_enc", "cluster",
    "family_enc", "city_enc", "state_enc",
    "store_family_mean_sales",

    # Promo
    "onpromotion", "promo_lag_7", "promo_roll_7_mean",

    # Oil (16-day embargo)
    "oil_lag_16", "oil_lag_21", "oil_lag_28",

    # Holiday / event flags
    "is_holiday", "is_holiday_national", "is_holiday_regional",
    "is_holiday_local", "is_event", "is_earthquake", "is_work_day",

    # Sales lags (16-day embargo)
    "sales_lag_16", "sales_lag_21", "sales_lag_28",

    # Rolling stats (shifted by 16 days)
    "sales_roll_7_mean", "sales_roll_28_mean",
]

TARGET = "sales"



# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------
def rmsle(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Log Error. Predictions clipped to 0 (no negatives)."""
    p = np.clip(predicted, 0, None)
    return float(np.sqrt(np.mean((np.log1p(p) - np.log1p(actual)) ** 2)))


# ---------------------------------------------------------------------------
# Data loading + splitting
# ---------------------------------------------------------------------------
def load_and_split(
    path: Path,
)-> tuple[
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
]:
    """Load features parquet, apply QC mask, split into train/val."""
    print(f"[TRAIN] Loading {path.name} ...")
    df = pd.read_parquet(path)

    # Exclude QC-flagged rows from the training window only
    # (zero series and pre-first-sale rows add noise with zero targets)
    qc_mask = (df["qc_zero_series"].fillna(0) == 0) & (df["qc_before_first_sale"].fillna(0) == 0)

    train_mask = (df["date"] <= pd.Timestamp(config.TRAIN_END)) & qc_mask
    val_mask   = (df["date"] >= pd.Timestamp(config.VAL_START)) & \
                 (df["date"] <= pd.Timestamp(config.VAL_END))

    train_df = df[train_mask].reset_index(drop=True)
    val_df   = df[val_mask].reset_index(drop=True)

    # Only keep feature columns that actually exist in the parquet
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]

    X_train = train_df[feat_cols].values.astype("float32")
    y_train = train_df[TARGET].values.astype("float64")
    X_val   = val_df[feat_cols].values.astype("float32")
    y_val   = val_df[TARGET].values.astype("float64")

    print(f"[TRAIN] Train rows: {len(X_train):,}  |  Val rows: {len(X_val):,}")
    print(f"[TRAIN] Features  : {len(feat_cols)}")
    return train_df, val_df, X_train, y_train, X_val, y_val, feat_cols


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
def build_lr_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   LinearRegression(n_jobs=-1)),
    ])


def build_rf_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   RandomForestRegressor(
            n_estimators=30,
            max_depth=20,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42
        )),
    ])


def build_xgb_pipeline(X_val, y_val) -> object:
    """XGBoost with early stopping — returns a plain XGBRegressor (no Pipeline wrapper)."""
    try:
        from xgboost import XGBRegressor
    except ImportError:
        raise ImportError("xgboost is missing. Run:  pip install xgboost")

    # Impute NaNs for XGB (it can handle them natively but we stay consistent)
    imp = SimpleImputer(strategy="median")

    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        eval_metric="rmse",
        early_stopping_rounds=30,
        verbosity=0,
    )
    return imp, model


# ---------------------------------------------------------------------------
# Feature importance (best-effort for each model type)
# ---------------------------------------------------------------------------
def get_importances(pipeline, feat_cols: list[str]) -> dict[str, float] | None:
    try:
        if hasattr(pipeline, "named_steps"):
            m = pipeline.named_steps.get("model")
        else:
            m = pipeline
        if hasattr(m, "feature_importances_"):
            fi = m.feature_importances_
            return dict(sorted(zip(feat_cols, fi), key=lambda x: -x[1]))
        if hasattr(m, "coef_"):
            # For LR: use absolute coefficient after scaling
            fi = np.abs(m.coef_)
            return dict(sorted(zip(feat_cols, fi), key=lambda x: -x[1]))
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------
def try_log_mlflow(run_name: str, params: dict, metrics: dict,
                   importances: dict | None) -> None:
    try:
        import mlflow
        mlflow.set_experiment("StockWise-AI")
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            if importances:
                top10 = dict(list(importances.items())[:10])
                mlflow.log_params({f"fi_{k}": f"{v:.4f}" for k, v in top10.items()})
        print(f"  MLflow run '{run_name}' logged.")
    except Exception as e:
        print(f"  MLflow logging skipped: {e}")


# ---------------------------------------------------------------------------
# Train one model, evaluate, return result dict
# ---------------------------------------------------------------------------
def train_and_eval(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feat_cols: list[str],
    use_mlflow: bool,
) -> dict:

    
    print(f"\n{'='*60}")
    print(f"[TRAIN] Training: {name}")
    print(f"{'='*60}")
    t0 = time.time()

    if name == "LinearRegression":
        pipeline = build_lr_pipeline()
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_val)
        fitted = pipeline

    elif name == "RandomForest":
        pipeline = build_rf_pipeline()
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_val)
        fitted = pipeline

    elif name == "XGBoost":
        imp, model = build_xgb_pipeline(X_val, y_val)
        X_tr_imp = imp.fit_transform(X_train)
        X_va_imp = imp.transform(X_val)
        # XGB can use log1p target to improve RMSLE directly
        model.fit(
            X_tr_imp, np.log1p(y_train),
            eval_set=[(X_va_imp, np.log1p(y_val))],
            verbose=False,
        )
        preds = np.expm1(model.predict(X_va_imp))
        fitted = (imp, model)
    else:
        raise ValueError(f"Unknown model: {name}")

    elapsed = time.time() - t0
    score = rmsle(y_val, preds)

    print(f"  RMSLE : {score:.5f}")
    print(f"  Time  : {elapsed:.1f}s")

    importances = get_importances(fitted if name != "XGBoost" else fitted[1], feat_cols)

    if use_mlflow:
        try_log_mlflow(
            run_name=name,
            params={"model": name},
            metrics={"val_rmsle": score, "train_time_s": elapsed},
            importances=importances,
        )

    return {
        "name": name,
        "rmsle": score,
        "time_s": elapsed,
        "preds": preds,
        "fitted": fitted,
        "importances": importances,
    }

def save_comparison_artifacts(
    results: list[dict],
    val_df: pd.DataFrame,
    feat_cols: list[str],
) -> None:
    """Save model predictions and metadata for compare_models.py."""

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save predictions from every successfully trained model
    base = val_df[
        ["id", "date", "store_nbr", "family", "sales"]
    ].copy()

    base = base.rename(columns={"sales": "actual_sales"})

    frames = []

    for result in results:
        model_df = base.copy()
        model_df["model"] = result["name"]
        model_df["predicted_sales"] = np.clip(
            result["preds"], 0, None
        )
        frames.append(model_df)

    comparison_df = pd.concat(frames, ignore_index=True)

    comparison_df.to_parquet(
        config.MODEL_COMPARISON_RESULTS_PATH,
        index=False,
    )

    # Save model metrics and feature importances
    best = min(results, key=lambda r: r["rmsle"])

    metadata = {
        "train_end": str(config.TRAIN_END),
        "val_start": str(config.VAL_START),
        "val_end": str(config.VAL_END),
        "features": feat_cols,
        "best_model": best["name"],
        "models": [
            {
                "name": r["name"],
                "rmsle": float(r["rmsle"]),
                "time_s": float(r["time_s"]),
                "importances": {
                    k: float(v)
                    for k, v in (r["importances"] or {}).items()
                },
            }
            for r in results
        ],
    }

    config.MODEL_COMPARISON_METADATA_PATH.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Saved: {config.MODEL_COMPARISON_RESULTS_PATH}")
    print(f"Saved: {config.MODEL_COMPARISON_METADATA_PATH}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "lr":  "LinearRegression",
    "rf":  "RandomForest",
    "xgb": "XGBoost",
}


def main(argv: list[str] | None = None) -> int:
    setup_console()
    ap = argparse.ArgumentParser(description="StockWise AI: train and compare models.")
    ap.add_argument(
        "--models", nargs="+", choices=list(MODEL_REGISTRY.keys()),
        default=list(MODEL_REGISTRY.keys()),
        help="Which models to run (default: all three)",
    )
    ap.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging")
    args = ap.parse_args(argv)

    if not config.FEATURES_TRAIN_PATH.exists():
        print("train_features.parquet not found. Run first:  python src/feature_engineering.py")
        return 1

    try:
        import pyarrow  # noqa: F401
    except ImportError:
        print("pyarrow is missing. Run:  pip install pyarrow")
        return 1

    use_mlflow = not args.no_mlflow
    model_names = [MODEL_REGISTRY[k] for k in args.models]

    train_df, val_df, X_train, y_train, X_val, y_val, feat_cols = load_and_split(
        config.FEATURES_TRAIN_PATH
    )

    results = []
    for mname in model_names:
        try:
            res = train_and_eval(mname, X_train, y_train, X_val, y_val,
                                 feat_cols, use_mlflow)
            results.append(res)
        except Exception as e:
            print(f"  [ERROR] {mname} failed: {e}")

    if not results:
        print("No models completed successfully.")
        return 1

    # Best model
    best = min(results, key=lambda x: x["rmsle"])
    print(f"\n{'='*60}")
    print(f"Best model : {best['name']}  (RMSLE = {best['rmsle']:.5f})")
    print(f"{'='*60}")

    # Save best predictions
    # Save best model predictions
    pred_df = val_df[
        ["id", "date", "store_nbr", "family", "sales"]
    ].copy()
    
    pred_df["predicted_sales"] = np.clip(best["preds"], 0, None)
    
    pred_df["rmsle_row"] = (
        np.log1p(pred_df["predicted_sales"])
        - np.log1p(pred_df["sales"])
    ) ** 2
    
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    pred_df.to_parquet(
        config.BEST_PREDICTIONS_PATH,
        index=False,
    )
    
    # Save predictions and metadata for all models
    save_comparison_artifacts(results, val_df, feat_cols)
    
    print(f"Saved: {config.BEST_PREDICTIONS_PATH}")

    # Save best model
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best["fitted"], "features": feat_cols,
                 "name": best["name"]}, config.BEST_MODEL_PATH)
    print(f"Saved: {config.BEST_MODEL_PATH}")

    # Write markdown report
    
    print("\n✓ Training completed successfully!")
    print(f"  Comparison data    → {config.MODEL_COMPARISON_RESULTS_PATH}")
    print(f"  Metadata           → {config.MODEL_COMPARISON_METADATA_PATH}")
    print(f"  Best predictions   → {config.BEST_PREDICTIONS_PATH}")
    print(f"  Best model         → {config.BEST_MODEL_PATH}")
    if use_mlflow:
        print("  MLflow UI          →  mlflow ui   (run in project root)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

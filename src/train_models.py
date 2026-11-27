"""
STEP 8: Train and compare Linear Regression, Random Forest,
XGBoost and XGBoost L1 on the VALIDATION set.

Run from project root:
    python src/train_models.py
    python src/train_models.py --models lr rf
    python src/train_models.py --models xgb xgb_l1
    python src/train_models.py --no-mlflow

Main metric: WAPE (lower is better).
Forecast Score = 100 - WAPE (higher is better).
NOTE: Forecast Score is NOT classification accuracy.

Time-based split only. No shuffling and no test-set usage.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

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


# ============================================================
# 1. FEATURES
# ============================================================

FEATURE_COLS = [
    # Calendar
    "day_of_week",
    "day_of_month",
    "week_of_year",
    "month",
    "quarter",
    "year",
    "is_weekend",
    "days_to_month_end",
    "is_payday",
    "days_to_next_holiday",
    "days_since_last_holiday",

    # Label codes
    "store_nbr",
    "store_type_enc",
    "cluster",
    "family_enc",
    "city_enc",
    "state_enc",

    # Series size
    "store_family_mean_sales",

    # Promotions and oil
    "onpromotion",
    "promo_lag_7",
    "promo_roll_7_mean",
    "oil_lag_16",
    "oil_lag_21",
    "oil_lag_28",

    # Holidays and events
    "is_holiday",
    "is_holiday_national",
    "is_holiday_regional",
    "is_holiday_local",
    "is_event",
    "is_earthquake",
    "is_work_day",

    # Sales history
    "sales_lag_16",
    "sales_lag_21",
    "sales_lag_28",
    "sales_roll_7_mean",
    "sales_roll_28_mean",
]

LABEL_CODE_COLS = [
    "store_nbr",
    "store_type_enc",
    "cluster",
    "family_enc",
    "city_enc",
    "state_enc",
]

LINEAR_FEATURE_COLS = [
    c for c in FEATURE_COLS
    if c not in LABEL_CODE_COLS
]

TARGET = "sales"

ID_COLS = [
    "id",
    "date",
    "store_nbr",
    "family",
]

QC_TRAIN_FILTERS = [
    "qc_zero_series",
    "qc_before_first_sale",
    "qc_sales_suspect",
]

EARLY_STOP_DAYS = 28

MLFLOW_URI = (
    f"sqlite:///{(config.PROJECT_ROOT / 'mlflow.db').as_posix()}"
)

MLFLOW_EXPERIMENT = "stockwise-ai"

MODEL_REGISTRY = {
    "lr": "LinearRegression",
    "rf": "RandomForest",
    "xgb": "XGBoost",
    "xgb_l1": "XGBoost_L1",
}

DEFAULT_MODELS = ["lr", "rf", "xgb"]

OFFICIAL_MODELS = (
    "LinearRegression",
    "RandomForest",
    "XGBoost",
)


# ============================================================
# 2. EVALUATION METRICS
# ============================================================

def mae(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - pred)))


def rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(
        np.sqrt(np.mean((actual - pred) ** 2))
    )


def wape(actual: np.ndarray, pred: np.ndarray) -> float:
    """
    Weighted Absolute Percentage Error.

    WAPE = 100 * sum(abs(actual - predicted)) / sum(actual)

    Main project metric. Lower is better.
    """

    total = float(np.sum(actual))

    if total <= 0:
        return float("nan")

    return float(
        100.0 * np.sum(np.abs(actual - pred)) / total
    )


def all_metrics(
    actual: np.ndarray,
    pred: np.ndarray
) -> dict[str, float]:

    # All models are evaluated after clipping predictions to zero.
    p = np.clip(pred, 0, None)

    scores = {
        "val_mae": mae(actual, p),
        "val_rmse": rmse(actual, p),
        "val_wape": wape(actual, p),
    }

    # Only the three metrics the team agreed on: MAE, RMSE, WAPE.
    return scores


# ============================================================
# 3. TIME-BASED DATA SPLIT
# ============================================================

def split_dates():

    train_end = pd.Timestamp(config.TRAIN_END)

    val_start = pd.Timestamp(config.VAL_START)

    test_start = pd.Timestamp(
        getattr(config, "TEST_START", "2017-07-31")
    )

    val_end = pd.Timestamp(config.VAL_END)

    # Never allow validation to overlap the test period.
    if val_end >= test_start:

        print(
            f"NOTE: VAL_END reaches test period. "
            f"Using {(test_start - pd.Timedelta(days=1)).date()}."
        )

        val_end = test_start - pd.Timedelta(days=1)

    return train_end, val_start, val_end


def load_and_split(path: Path):

    import pyarrow.parquet as pq

    available = set(
        pq.read_schema(path).names
    )

    features = [
        c for c in FEATURE_COLS
        if c in available
    ]

    missing = [
        c for c in FEATURE_COLS
        if c not in available
    ]

    if missing:

        print(
            f"NOTE: Missing features: {missing}\n"
            "Run feature_engineering.py to generate them."
        )

    filters = [
        c for c in QC_TRAIN_FILTERS
        if c in available
    ]

    needed = list(
        dict.fromkeys(
            ID_COLS + [TARGET] + features + filters
        )
    )

    print(f"\n[TRAIN] Loading {path.name} ...")

    df = pd.read_parquet(
        path,
        columns=needed
    )

    df["date"] = pd.to_datetime(df["date"])

    train_end, val_start, val_end = split_dates()

    keep = pd.Series(
        True,
        index=df.index
    )

    # Apply QC filters ONLY to training data.
    for f in filters:

        keep &= df[f].fillna(0) == 0

    train_df = df[
        (df["date"] <= train_end) & keep
    ].reset_index(drop=True)

    # Validation is never filtered.
    val_df = df[
        (df["date"] >= val_start) &
        (df["date"] <= val_end)
    ].reset_index(drop=True)

    if train_df.empty or val_df.empty:

        raise SystemExit(
            f"Empty split: "
            f"train={len(train_df):,}, "
            f"validation={len(val_df):,}"
        )

    if (
        train_df[TARGET].isna().any()
        or val_df[TARGET].isna().any()
    ):

        raise SystemExit(
            "Missing sales values in training/validation."
        )

    print("\n" + "=" * 60)
    print("DATA SPLIT")
    print("=" * 60)

    print(
        f"Training rows    : {len(train_df):,}"
    )

    print(
        f"Validation rows  : {len(val_df):,}"
    )

    print(
        f"Training ends    : {train_end.date()}"
    )

    print(
        f"Validation       : {val_start.date()} "
        f"-> {val_end.date()}"
    )

    print(
        f"Features         : {len(features)}"
    )

    print(
        f"QC filters       : {filters}"
    )

    return train_df, val_df, features


def matrix(
    df: pd.DataFrame,
    cols: list[str]
) -> np.ndarray:

    return df[cols].to_numpy(
        dtype="float32"
    )


# ============================================================
# 4. MODEL BUILDERS
# ============================================================

def build_lr() -> Pipeline:

    return Pipeline([

        (
            "imputer",
            SimpleImputer(strategy="median")
        ),

        (
            "scaler",
            StandardScaler()
        ),

        (
            "model",
            LinearRegression(n_jobs=-1)
        ),

    ])


def build_rf() -> Pipeline:

    return Pipeline([

        (
            "imputer",
            SimpleImputer(strategy="median")
        ),

        (
            "model",
            RandomForestRegressor(

                # Increased from 30 to 150 trees
                n_estimators=150,

                max_depth=20,

                min_samples_leaf=5,

                max_features="sqrt",

                n_jobs=-1,

                random_state=42

            )
        ),

    ])


def xgb_model(
    n_estimators: int,
    objective: str,
    eval_metric: str,
    early_stopping: bool
):

    try:

        from xgboost import XGBRegressor

    except ImportError:

        raise ImportError(
            "xgboost is missing. Run: pip install xgboost"
        )

    extra = (
        {"early_stopping_rounds": 30}
        if early_stopping
        else {}
    )

    return XGBRegressor(

        n_estimators=n_estimators,

        learning_rate=0.05,

        max_depth=7,

        subsample=0.8,

        colsample_bytree=0.8,

        min_child_weight=10,

        tree_method="hist",

        random_state=42,

        n_jobs=-1,

        objective=objective,

        eval_metric=eval_metric,

        verbosity=0,

        **extra

    )


# ============================================================
# 5. XGBOOST TRAINING
# ============================================================

def fit_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    dates: pd.Series,
    log_target: bool,
    objective: str,
    eval_metric: str
):

    """
    Honest early stopping.

    1. Use the last 28 days of TRAINING as an internal holdout.
    2. Select the number of trees using that holdout.
    3. Refit using the complete training period.

    Validation is NOT used for early stopping.
    """

    imputer = SimpleImputer(
        strategy="median"
    )

    Xi = imputer.fit_transform(X)

    target = (
        np.log1p(y)
        if log_target
        else y
    )

    cut = (
        pd.Timestamp(config.TRAIN_END)
        - pd.Timedelta(days=EARLY_STOP_DAYS)
    )

    core = (
        dates <= cut
    ).to_numpy()

    hold = ~core

    info = {}

    if (
        core.sum() >= 1000
        and hold.sum() >= 1000
    ):

        print(
            "\n[XGBOOST] Running internal early stopping..."
        )

        probe = xgb_model(
            1000,
            objective,
            eval_metric,
            early_stopping=True
        )

        probe.fit(

            Xi[core],

            target[core],

            eval_set=[
                (Xi[hold], target[hold])
            ],

            verbose=False

        )

        n_trees = int(
            probe.best_iteration
        ) + 1

        info["early_stopping"] = (
            f"last {EARLY_STOP_DAYS} days of training"
        )

    else:

        n_trees = 300

        info["early_stopping"] = (
            "not enough rows, fixed 300 trees"
        )

    print(
        f"[XGBOOST] Selected trees: {n_trees}"
    )

    # Refit on ALL training data.
    final = xgb_model(
        n_trees,
        objective,
        eval_metric,
        early_stopping=False
    )

    final.fit(
        Xi,
        target
    )

    info["n_trees"] = n_trees

    return imputer, final, info


# ============================================================
# 6. FEATURE IMPORTANCE
# ============================================================

def get_importances(
    model,
    names: list[str]
):

    try:

        inner = (
            model.named_steps["model"]
            if hasattr(model, "named_steps")
            else model
        )

        if hasattr(inner, "feature_importances_"):

            values = inner.feature_importances_

        elif hasattr(inner, "coef_"):

            values = np.abs(inner.coef_)

        else:

            return None

        return dict(
            sorted(
                zip(
                    names,
                    map(float, values)
                ),
                key=lambda kv: -kv[1]
            )
        )

    except Exception:

        return None


# ============================================================
# 7. TRAIN ONE MODEL
# ============================================================

def train_one(
    name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: list[str]
) -> dict:

    print("\n" + "=" * 65)
    print(f"TRAINING: {name}")
    print("=" * 65)

    # Linear Regression does not use ordinal label codes.
    if name == "LinearRegression":

        cols = [
            c for c in LINEAR_FEATURE_COLS
            if c in features
        ]

    else:

        cols = features

    X_tr = matrix(
        train_df,
        cols
    )

    y_tr = train_df[TARGET].to_numpy(
        dtype="float64"
    )

    X_va = matrix(
        val_df,
        cols
    )

    y_va = val_df[TARGET].to_numpy(
        dtype="float64"
    )

    t0 = time.time()

    params = {
        "clues": len(cols),
        "target": "sales"
    }

    # --------------------------------------------------------
    # Linear Regression
    # --------------------------------------------------------

    if name == "LinearRegression":

        print("[LR] Fitting Linear Regression...")

        fitted = build_lr().fit(
            X_tr,
            y_tr
        )

        pred = fitted.predict(X_va)

        saved = fitted

        transform = "none"

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    elif name == "RandomForest":

        print(
            "[RF] Fitting Random Forest with 150 trees..."
        )

        fitted = build_rf().fit(
            X_tr,
            y_tr
        )

        pred = fitted.predict(X_va)

        saved = fitted

        transform = "none"

        params.update({

            "n_estimators": 150,

            "max_depth": 20,

            "min_samples_leaf": 5,

            "max_features": "sqrt",

        })

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    elif name == "XGBoost":

        print(
            "[XGBOOST] Training with log1p(sales)..."
        )

        imp, model, info = fit_xgboost(

            X_tr,

            y_tr,

            train_df["date"],

            True,

            "reg:squarederror",

            "rmse"

        )

        pred = np.expm1(
            model.predict(
                imp.transform(X_va)
            )
        )

        fitted = model

        saved = (
            imp,
            model
        )

        transform = "log1p"

        params.update(

            info,

            target="log1p(sales)",

            objective="squared error on log1p(sales)"

        )

    # --------------------------------------------------------
    # XGBoost L1 experiment
    # --------------------------------------------------------

    elif name == "XGBoost_L1":

        print(
            "[XGBOOST L1] Training with absolute error..."
        )

        imp, model, info = fit_xgboost(

            X_tr,

            y_tr,

            train_df["date"],

            False,

            "reg:absoluteerror",

            "mae"

        )

        pred = model.predict(
            imp.transform(X_va)
        )

        fitted = model

        saved = (
            imp,
            model
        )

        transform = "none"

        params.update(

            info,

            objective="absolute error on sales"

        )

    else:

        raise ValueError(
            f"Unknown model: {name}"
        )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    elapsed = time.time() - t0

    pred = np.clip(
        pred,
        0,
        None
    )

    scores = all_metrics(
        y_va,
        pred
    )

    print("\n" + "-" * 60)
    print(f"RESULTS: {name}")
    print("-" * 60)

    print(
        f"WAPE           : "
        f"{scores['val_wape']:.2f}%"
    )

    print(
        f"MAE            : "
        f"{scores['val_mae']:.3f}"
    )

    print(
        f"RMSE           : "
        f"{scores['val_rmse']:.3f}"
    )

    print(
        f"Training time  : "
        f"{elapsed:.1f} seconds"
    )

    print("-" * 60)

    return {

        "name": name,

        "kind": (
            "model"
            if name in OFFICIAL_MODELS
            else "experiment"
        ),

        "preds": pred,

        "saved": saved,

        "target_transform": transform,

        "features": cols,

        "scores": scores,

        "time_s": elapsed,

        "importances": get_importances(
            fitted,
            cols
        ),

        "params": params

    }


# ============================================================
# 8. MLFLOW
# ============================================================

def log_mlflow(
    result: dict,
    shared: dict
) -> None:

    try:

        import mlflow

        mlflow.set_tracking_uri(
            MLFLOW_URI
        )

        mlflow.set_experiment(
            MLFLOW_EXPERIMENT
        )

        with mlflow.start_run(
            run_name=result["name"]
        ):

            mlflow.log_params({

                "model": result["name"],

                "kind": result["kind"],

                **shared,

                **result["params"]

            })

            mlflow.log_metrics({

                **result["scores"],

                "train_time_s": result["time_s"]

            })

            if result["importances"]:

                mlflow.log_params({

                    f"fi_{k}": f"{v:.4f}"

                    for k, v in list(
                        result["importances"].items()
                    )[:10]

                })

        print(
            f"[MLFLOW] Logged {result['name']}"
        )

    except Exception as exc:

        print(
            f"[MLFLOW] Logging skipped: {exc}"
        )


# ============================================================
# 9. SAVE OUTPUTS
# ============================================================

def save_artifacts(
    results: list[dict],
    val_df: pd.DataFrame,
    shared: dict
) -> dict:

    config.REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    base = val_df[
        [
            "id",
            "date",
            "store_nbr",
            "family",
            TARGET
        ]
    ].rename(
        columns={
            TARGET: "actual_sales"
        }
    )

    # --------------------------------------------------------
    # Save predictions of all models
    # --------------------------------------------------------

    frames = []

    for r in results:

        f = base.copy()

        f["model"] = r["name"]

        f["predicted_sales"] = r["preds"]

        frames.append(f)

    pd.concat(
        frames,
        ignore_index=True
    ).to_parquet(
        config.MODEL_COMPARISON_RESULTS_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Select best OFFICIAL model using WAPE
    # --------------------------------------------------------

    official = [
        r for r in results
        if r["kind"] == "model"
    ]

    candidates = official if official else results

    best = min(
        candidates,
        key=lambda r: r["scores"]["val_wape"]
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata = {

        **{
            k: v
            for k, v in shared.items()
            if k in (
                "train_end",
                "val_start",
                "val_end"
            )
        },

        "features": list(
            dict.fromkeys(
                c
                for r in results
                for c in r["features"]
            )
        ),

        "best_model": best["name"],

        "best_model_rule": (
            "lowest validation WAPE among official models"
        ),

        "main_metric": "WAPE",

        "models": [

            {

                "name": r["name"],

                "kind": r["kind"],

                "scores": {

                    k: float(v)

                    for k, v in r["scores"].items()

                },

                "time_s": float(
                    r["time_s"]
                ),

                "features": r["features"],

                "params": r["params"],

                "importances": {

                    k: float(v)

                    for k, v in (
                        r["importances"] or {}
                    ).items()

                }

            }

            for r in results

        ]

    }

    config.MODEL_COMPARISON_METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2
        ),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Save best model predictions
    # --------------------------------------------------------

    pred = base.rename(
        columns={
            "actual_sales": "sales"
        }
    ).copy()

    pred["predicted_sales"] = best["preds"]

    pred["abs_error"] = (
        pred["sales"] -
        pred["predicted_sales"]
    ).abs()

    pred.to_parquet(
        config.BEST_PREDICTIONS_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Save best model object
    # --------------------------------------------------------

    config.MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(

        {

            "model": best["saved"],

            "features": best["features"],

            "name": best["name"],

            "target_transform": best["target_transform"]

        },

        config.BEST_MODEL_PATH

    )

    print("\n[FILES] Saved successfully:")

    print(
        config.MODEL_COMPARISON_RESULTS_PATH
    )

    print(
        config.MODEL_COMPARISON_METADATA_PATH
    )

    print(
        config.BEST_PREDICTIONS_PATH
    )

    print(
        config.BEST_MODEL_PATH
    )

    return {
        "best": best["name"]
    }


# ============================================================
# 10. MAIN
# ============================================================

def main(
    argv: list[str] | None = None
) -> int:

    setup_console()

    ap = argparse.ArgumentParser(

        description=(
            "StockWise AI: train and compare forecasting models."
        )

    )

    ap.add_argument(

        "--models",

        nargs="+",

        choices=list(MODEL_REGISTRY),

        default=DEFAULT_MODELS,

        help=(
            "Choose models: lr rf xgb xgb_l1"
        )

    )

    ap.add_argument(

        "--no-mlflow",

        action="store_true",

        help="Skip MLflow logging"

    )

    args = ap.parse_args(argv)

    # --------------------------------------------------------
    # Check files and dependencies
    # --------------------------------------------------------

    if not config.FEATURES_TRAIN_PATH.exists():

        print(
            "[ERROR] train_features.parquet not found."
        )

        print(
            "Run: python src/feature_engineering.py"
        )

        return 1

    try:

        import pyarrow  # noqa: F401

    except ImportError:

        print(
            "[ERROR] pyarrow is missing."
        )

        print(
            "Run: pip install pyarrow"
        )

        return 1

    # --------------------------------------------------------
    # Load and split data
    # --------------------------------------------------------

    train_df, val_df, features = load_and_split(
        config.FEATURES_TRAIN_PATH
    )

    train_end, val_start, val_end = split_dates()

    shared = {

        "train_end": str(
            train_end.date()
        ),

        "val_start": str(
            val_start.date()
        ),

        "val_end": str(
            val_end.date()
        ),

        "n_train_rows": len(train_df),

        "n_val_rows": len(val_df),

        "train_filter": (
            "no zero series, "
            "no rows before first sale, "
            "no suspected data errors"
        )

    }

    # --------------------------------------------------------
    # Train selected models
    # --------------------------------------------------------

    results = []

    for key in args.models:

        name = MODEL_REGISTRY[key]

        try:

            result = train_one(

                name,

                train_df,

                val_df,

                features

            )

            results.append(result)

            if not args.no_mlflow:

                log_mlflow(
                    result,
                    shared
                )

        except Exception as exc:

            print(
                f"\n[ERROR] {name} failed:"
            )

            print(
                f"{type(exc).__name__}: {exc}"
            )

    if not results:

        print(
            "\n[ERROR] No model finished successfully."
        )

        return 1

    # --------------------------------------------------------
    # Save all artifacts
    # --------------------------------------------------------

    summary = save_artifacts(

        results,

        val_df,

        shared

    )

    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    print("\n")

    print("=" * 90)

    print("FINAL MODEL COMPARISON")

    print("=" * 90)

    print(
        "Lower WAPE / MAE / RMSE / RMSLE is better."
    )

    print(
        "Higher Forecast Score is better."
    )

    print("-" * 90)

    sorted_results = sorted(

        results,

        key=lambda r: r["scores"]["val_wape"]

    )

    for r in sorted_results:

        s = r["scores"]

        print(

            f"{r['name']:<18}"

            f" WAPE: {s['val_wape']:7.2f}%"

            f" | MAE: {s['val_mae']:10.3f}"

            f" | RMSE: {s['val_rmse']:10.3f}"

            f" | {r['kind']}"

        )

    print("-" * 90)

    print(
        f"\nBEST OFFICIAL MODEL: {summary['best']}"
    )

    print(
        "\nNext step:"
    )

    print(
        "python src/final_comparison.py"
    )

    print("\nTraining completed successfully.")

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )

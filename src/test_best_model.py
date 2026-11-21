"""LAST STEP OF LEVEL 1: score the chosen model ONCE on the test period.

Run from the project root (after python src/run_pipeline.py):
    python src/test_best_model.py

Why this is separate from everything else
-----------------------------------------
The test period (TEST_START .. TEST_END = 2017-07-31 .. 2017-08-15) was never
used to build, tune or choose any model. The best model was chosen on validation
(lowest validation WAPE, see reports/model_comparison.md). This script gives the
honest final number: how good is that model on days it has never seen?

What it does (the standard final step)
--------------------------------------
1. Read which model won (models/best_model.pkl) and its exact clues.
2. Retrain the SAME model, with the SAME settings, on training + validation
   (everything up to VAL_END = 2017-07-30). This is what a shop would do:
   standing on 30 July, use all the history it has, forecast the next 16 days.
   (Sales clues are at least 16 days old, so a forecast for 15 Aug uses sales
   up to 30 Jul at the latest: exactly "standing on 30 July".)
3. Predict the 16 test days and score them ONCE: MAE, RMSE, WAPE.
4. Log the result to MLflow (metrics test_mae, test_rmse, test_wape) and save
   reports/test_result.md / .json.

Running it again would mean "looking at the test period twice", so the script
refuses if reports/test_result.json already exists (use --force only if the
model or the data really changed).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd

import config
from data_io import setup_console
import train_models as tm

RESULT_JSON = config.REPORTS_DIR / "test_result.json"
RESULT_MD = config.REPORTS_DIR / "test_result.md"
NAIVE_METRICS = config.REPORTS_DIR / "naive_baseline_metrics.csv"


def fit_same_model(name: str, X: np.ndarray, y: np.ndarray, dates: pd.Series):
    """Build and fit the winning model with exactly the settings used in train_models.py."""
    if name == "RandomForest":
        return tm.build_rf().fit(X, y), "none"
    if name == "LinearRegression":
        return tm.build_lr().fit(X, y), "none"
    if name == "XGBoost":
        imputer, model, _ = tm.fit_xgboost(X, y, dates, log_target=True,
                                           objective="reg:squarederror", eval_metric="rmse")
        return (imputer, model), "log1p"
    raise SystemExit(f"Unknown model '{name}': add it to fit_same_model().")


def predict(model, transform: str, X: np.ndarray) -> np.ndarray:
    if isinstance(model, tuple):                       # XGBoost: (imputer, model)
        imputer, booster = model
        pred = booster.predict(imputer.transform(X))
    else:
        pred = model.predict(X)
    if transform == "log1p":
        pred = np.expm1(pred)
    return np.clip(pred, 0, None)                       # sales cannot be negative


def main(argv: list[str] | None = None) -> int:
    setup_console()
    ap = argparse.ArgumentParser(description="Score the chosen model once on the test period.")
    ap.add_argument("--force", action="store_true", help="run again although a test result exists")
    ap.add_argument("--no-mlflow", action="store_true", help="do not log to MLflow")
    args = ap.parse_args(argv)

    if RESULT_JSON.exists() and not args.force:
        print(f"{RESULT_JSON.name} already exists: the test period has been scored once already.")
        print("Read reports/test_result.md. Use --force only if the model or the data changed.")
        return 1
    for path in (config.BEST_MODEL_PATH, config.FEATURES_TRAIN_PATH):
        if not path.exists():
            print(f"{path.name} not found. Run first:  python src/run_pipeline.py")
            return 1

    pack = joblib.load(config.BEST_MODEL_PATH)
    name, features = pack["name"], pack["features"]
    print(f"Chosen model (lowest validation WAPE): {name}, {len(features)} clues")

    import pyarrow.parquet as pq
    available = set(pq.read_schema(config.FEATURES_TRAIN_PATH).names)
    filters = [c for c in tm.QC_TRAIN_FILTERS if c in available]
    columns = list(dict.fromkeys(["date", "store_nbr", "family", tm.TARGET] + features + filters))
    df = pd.read_parquet(config.FEATURES_TRAIN_PATH, columns=columns)
    df["date"] = pd.to_datetime(df["date"])

    fit_end = pd.Timestamp(config.VAL_END)
    test_start, test_end = pd.Timestamp(config.TEST_START), pd.Timestamp(config.TEST_END)

    keep = pd.Series(True, index=df.index)
    for f in filters:                                   # QC filters: TRAINING rows only
        keep &= df[f].fillna(0) == 0
    fit_df = df[(df["date"] <= fit_end) & keep]
    test_df = df[(df["date"] >= test_start) & (df["date"] <= test_end)]   # never filtered

    print(f"Retrain on : {fit_df['date'].min().date()} -> {fit_end.date()}  ({len(fit_df):,} rows)")
    print(f"Test period: {test_start.date()} -> {test_end.date()}  ({len(test_df):,} rows, "
          f"{test_df['date'].nunique()} days)")

    t0 = time.time()
    model, transform = fit_same_model(name, tm.matrix(fit_df, features),
                                      fit_df[tm.TARGET].to_numpy(dtype="float64"), fit_df["date"])
    fit_seconds = time.time() - t0

    actual = test_df[tm.TARGET].to_numpy(dtype="float64")
    pred = predict(model, transform, tm.matrix(test_df, features))
    scores = {"test_mae": tm.mae(actual, pred), "test_rmse": tm.rmse(actual, pred),
              "test_wape": tm.wape(actual, pred)}

    naive = pd.read_csv(NAIVE_METRICS, index_col=0) if NAIVE_METRICS.exists() else None
    naive_test_wape = float(naive.loc["test", "WAPE_%"]) if naive is not None else None
    comparison = pd.read_csv(config.REPORTS_DIR / "model_comparison.csv")
    val_wape = float(comparison.loc[comparison["model"] == name, "WAPE_%"].iloc[0])

    result = {
        "model": name,
        "trained_on": f"{fit_df['date'].min().date()} -> {fit_end.date()} (train + validation)",
        "test_period": f"{test_start.date()} -> {test_end.date()}",
        "test_rows": int(len(test_df)),
        **{k: round(v, 4) for k, v in scores.items()},
        "validation_wape": round(val_wape, 4),
        "naive_test_wape": naive_test_wape,
        "fit_seconds": round(fit_seconds, 1),
    }
    RESULT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")

    better = (f"{100 * (naive_test_wape - scores['test_wape']) / naive_test_wape:.1f}% less error than the naive "
              f"baseline on the same days ({naive_test_wape:.2f}%)") if naive_test_wape else ""
    lines = [
        "# Final test score (auto-generated by src/test_best_model.py)",
        "",
        f"Model: **{name}** (chosen because it had the lowest validation WAPE: {val_wape:.2f}%).",
        f"Retrained on {result['trained_on']}, then scored **once** on the test period "
        f"{result['test_period']} ({len(test_df):,} rows, never used before).",
        "",
        "| | MAE | RMSE | WAPE |",
        "|---|---|---|---|",
        f"| {name}, test | {scores['test_mae']:.2f} | {scores['test_rmse']:.2f} | **{scores['test_wape']:.2f}%** |",
    ]
    if naive is not None:
        lines.append(f"| Naive lag 21, test | {naive.loc['test', 'MAE']:.2f} | {naive.loc['test', 'RMSE']:.2f} | "
                     f"{naive_test_wape:.2f}% |")
    lines += ["", f"Read it as: on 16 days the model never saw, its total error is "
              f"{scores['test_wape']:.2f}% of all units sold; {better}.", ""]
    RESULT_MD.write_text("\n".join(lines), encoding="utf-8")

    if not args.no_mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri(tm.MLFLOW_URI)
            mlflow.set_experiment(tm.MLFLOW_EXPERIMENT)
            with mlflow.start_run(run_name=f"{name}_final_test"):
                mlflow.log_params({"model": name, "stage": "final test (scored once)",
                                   "trained_on": result["trained_on"], "test_period": result["test_period"]})
                mlflow.log_metrics({**scores, "validation_wape": val_wape})
            print("[MLFLOW] Logged the final test run")
        except Exception as exc:                                         # noqa: BLE001
            print(f"[MLFLOW] Logging skipped: {exc}")

    print("\n" + "\n".join(lines[5:9]))
    print(f"\nSaved: {RESULT_MD.relative_to(config.PROJECT_ROOT)}, {RESULT_JSON.relative_to(config.PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

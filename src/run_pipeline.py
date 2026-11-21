"""ONE COMMAND from raw data to the final comparison table.

Run from the project root:
    python src/run_pipeline.py                 # everything
    python src/run_pipeline.py --no-mlflow     # same, without MLflow logging
    python src/run_pipeline.py --models lr rf  # only some of Sara's models (lr rf xgb)
    python src/run_pipeline.py --from train_models      # restart from a step (the earlier steps already ran)
    python src/run_pipeline.py --skip leakage_check     # leave one step out
    python src/run_pipeline.py --inspect       # also look at the raw files first

The steps, in order (each one stops the pipeline if it fails):
    1. prepare_data         clean + merge the raw files, then PROVE the result (validation)
    2. feature_engineering  lags, rolling means, calendar and encodings
    3. leakage_check        PROVE that no feature uses information from the last 16 days
    4. train_models         Linear Regression, Random Forest, XGBoost
    5. train_polynomial     Polynomial Regression (degree 2)
    6. final_comparison     one fair table for all models + residual analysis + charts
The test slice (31 Jul - 15 Aug 2017) is never used by any of these steps.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import config
from data_io import missing_required_files, setup_console

STEPS = ["prepare_data", "feature_engineering", "leakage_check", "train_models_new", "train_polynomial", "final_comparison"]


def step_args(step: str, args: argparse.Namespace) -> list[str]:
    extra: list[str] = []
    if step in ("train_models_new", "train_polynomial") and args.no_mlflow:
        extra.append("--no-mlflow")
    if step == "train_models" and args.models:
        extra += ["--models", *args.models]
    return extra


def main(argv: list[str] | None = None) -> int:
    setup_console()
    ap = argparse.ArgumentParser(description="StockWise AI: run the whole pipeline.")
    ap.add_argument("--from", dest="start", choices=STEPS, help="start at this step")
    ap.add_argument("--skip", nargs="+", choices=STEPS, default=[], help="steps to leave out")
    ap.add_argument("--no-mlflow", action="store_true", help="do not log to MLflow")
    ap.add_argument("--models", nargs="+", choices=["lr", "rf", "xgb"], help="models for train_models.py")
    ap.add_argument("--inspect", action="store_true", help="also run inspect_data first")
    args = ap.parse_args(argv)

    missing = missing_required_files()
    if missing:
        print("Required files are missing from data/raw: " + ", ".join(p.name for p in missing))
        print("Download the Kaggle 'Store Sales - Time Series Forecasting' files into data/raw/ (see README).")
        return 1

    steps = (["inspect_data"] if args.inspect else []) + STEPS
    if args.start:
        steps = steps[steps.index(args.start):]
    steps = [s for s in steps if s not in args.skip]

    results: list[tuple[str, str, float]] = []
    t_all = time.time()
    for i, step in enumerate(steps, 1):
        script = HERE / f"{step}.py"
        if not script.exists():
            print(f"\n[{i}/{len(steps)}] {step}: {script.name} not found, skipped")
            results.append((step, "SKIPPED (file missing)", 0.0))
            continue
        print(f"\n{'#' * 78}\n[{i}/{len(steps)}] {step}\n{'#' * 78}", flush=True)
        t0 = time.time()
        code = subprocess.run([sys.executable, str(script), *step_args(step, args)], cwd=config.PROJECT_ROOT).returncode
        elapsed = time.time() - t0
        results.append((step, "OK" if code == 0 else f"FAILED (exit {code})", elapsed))
        if code != 0:
            break

    print(f"\n{'=' * 78}\nPIPELINE SUMMARY\n{'=' * 78}")
    for step, status, sec in results:
        print(f"  {step:<22} {status:<24} {sec:6.0f}s")
    failed = [s for s, st, _ in results if st.startswith("FAILED")]
    if failed:
        print(f"\nStopped at '{failed[0]}'. Fix it, then continue with:  python src/run_pipeline.py --from {failed[0]}")
        return 1
    print(f"\nAll done in {time.time() - t_all:.0f}s. Open reports/model_comparison.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
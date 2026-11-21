"""LEAKAGE CHECK: proves that no feature looks at information we would not have in real life.

Run from the project root (after prepare_data.py):
    python src/leakage_check.py

How the proof works (a "perturbation test"):
  1. Build the features normally.
  2. Destroy ALL sales and ALL oil prices after TRAIN_END (multiply them and add noise) and build the features
     again. (onpromotion is not touched: promotions are planned in advance, so the future values are known.)
  3. A feature of a row at date t may only use sales up to t - 16 days. So every feature of every row
     up to TRAIN_END + 15 days must be IDENTICAL in both runs. If even one differs, that feature
     peeked at the future (a leak) and the check FAILS.
  4. As a control, later rows MUST change, otherwise the test itself would be blind.
It also checks that the model feature lists never contain sales, ids, dates or qc_ flags.

Result: printed and saved to reports/leakage_report.txt  (exit code 1 if anything fails).
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import config
from data_io import Report, setup_console

HORIZON_DAYS = 16
FORBIDDEN_AS_FEATURE = {"sales", "sales_capped", "id", "date"} | set(config.QC_FLAGS)


def same(a: pd.Series, b: pd.Series) -> bool:
    x, y = a.to_numpy(), b.to_numpy()
    if x.dtype.kind in "fc" or y.dtype.kind in "fc":
        return bool(np.array_equal(x.astype("float64"), y.astype("float64"), equal_nan=True))
    return bool(np.array_equal(x, y))


def build_subset(fe, table: pd.DataFrame, safe_until: pd.Timestamp, new_cols: list[str] | None = None):
    """Build features, keep only what we compare, free the big table."""
    original = set(table.columns)
    feats, _ = fe.build_features(table, None)
    if new_cols is None:
        new_cols = [c for c in feats.columns if c not in original and c != "sales_capped"]
    feats = feats.sort_values("id").reset_index(drop=True)
    safe = feats["date"] <= safe_until
    inside = feats.loc[safe, ["id"] + new_cols].copy()
    later = feats.loc[~safe, ["id"] + [c for c in ("sales_lag_16",) if c in feats.columns]].copy()
    return inside, later, new_cols


def run_leakage_check(save_report: bool = True) -> bool:
    setup_console()
    rep = Report()
    fails = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal fails
        rep.add(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"   ({detail})" if detail else ""))
        fails += 0 if ok else 1

    rep.section("STOCKWISE AI - LEAKAGE CHECK (perturbation test)")
    if not config.TRAIN_CLEAN_PATH.exists():
        rep.add("train_clean.parquet not found. Run first:  python src/prepare_data.py")
        return False

    train_clean = pd.read_parquet(config.TRAIN_CLEAN_PATH)
    train_clean["date"] = pd.to_datetime(train_clean["date"])
    cut = pd.Timestamp(config.TRAIN_END)
    safe_until = cut + pd.Timedelta(days=HORIZON_DAYS - 1)
    fe = importlib.import_module("feature_engineering")
    rep.add(f"Future = everything after {cut.date()}.  Rows up to {safe_until.date()} must not change.")

    # ---- 1) normal run -------------------------------------------------------
    rep.add("Building features (normal) ...")
    inside_a, later_a, new_cols = build_subset(fe, train_clean.copy(), safe_until)

    # ---- 2) run with the future destroyed ------------------------------------
    rep.add("Building features (future sales and oil destroyed) ...")
    broken = train_clean.copy()
    future = broken["date"] > cut
    for col in ("sales", "sales_capped"):
        if col in broken.columns:
            broken.loc[future, col] = broken.loc[future, col] * 1000.0 + 12345.0
    if "oil_price" in broken.columns:
        broken.loc[future, "oil_price"] = broken.loc[future, "oil_price"] * 1.7 + 3.0
    del train_clean
    inside_b, later_b, _ = build_subset(fe, broken, safe_until, new_cols)
    del broken

    rep.section("Every engineered feature must be identical for rows up to the safe date")
    check("same rows in both runs", np.array_equal(inside_a["id"].to_numpy(), inside_b["id"].to_numpy()),
          f"{len(inside_a):,} rows compared")
    for col in new_cols:
        check(f"feature '{col}' does not use future information", same(inside_a[col], inside_b[col]))

    rep.section("Control: the test must be able to see a leak")
    if "sales_lag_16" in later_a.columns:
        check("rows AFTER the safe date do change (so the test is not blind)",
              not same(later_a["sales_lag_16"], later_b["sales_lag_16"]))

    rep.section("Model feature lists must not contain sales, ids, dates or qc_ flags")
    for module_name, attr_names in (("train_models_new", ["FEATURE_COLS"]), ("train_polynomial", ["FEATURES", "EXTRA_FEATURES"])):        
        try:
            module = importlib.import_module(module_name)
            cols = [c for a in attr_names for c in getattr(module, a)]
            bad = sorted(set(cols) & FORBIDDEN_AS_FEATURE)
            check(f"{module_name}: no forbidden column used as a feature", not bad,
                  f"{len(cols)} features" if not bad else f"forbidden: {bad}")
        except Exception as exc:  # the module may not exist yet
            rep.add(f"[SKIP] {module_name}: could not be loaded ({exc!r})")

    rep.section("FINAL RESULT")
    rep.add("NO LEAKAGE FOUND: every feature uses only information that is at least 16 days old." if fails == 0
            else f"LEAKAGE or problems found: {fails} check(s) failed. Fix the FAIL lines above.")
    if save_report:
        rep.save(config.REPORTS_DIR / "leakage_report.txt")
        print(f"\nSaved: {config.REPORTS_DIR / 'leakage_report.txt'}")
    return fails == 0


if __name__ == "__main__":
    raise SystemExit(0 if run_leakage_check() else 1)
"""STEP 7 (Adham): Feature engineering — build model-ready features from the cleaned data.

Run from the project root (after prepare_data.py):
    python src/feature_engineering.py
    python src/feature_engineering.py --no-test   # skip test set enrichment

Reads  : data/processed/train_clean.parquet  (and test_clean.parquet if present)
Writes : data/processed/train_features.parquet
         data/processed/test_features.parquet  (if test_clean exists)
         reports/feature_report.md

Feature groups
--------------
  Calendar  : day_of_week, day_of_month, week_of_year, month, quarter,
               is_weekend, days_to_month_end, is_payday
  Lag       : sales_lag_7/14/16/30  (date-merge, never .shift — Dec-25 safe)
  Rolling   : sales_roll_7/14/30_mean, sales_roll_7_std  (same approach)
  Promo     : onpromotion already present + promo_lag_7, promo_roll_7_mean
  Oil       : oil_price already present + oil_lag_7, oil_roll_7_mean
  Encoding  : store_family_mean_sales (global series mean, train only)
               store_type_enc (A→0 … E→4), cluster already numeric
               family_enc, city_enc, state_enc  (ordinal, train only)

CRITICAL: all statistics and encodings are fitted on the TRAINING window
(date <= TRAIN_END) only, then applied to the full table so the validation
and test rows get features without leaking any future information.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import config
from data_io import setup_console

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
LAG_DAYS   = [7, 14, 16, 30]
ROLL_WINS  = [7, 14, 30]
KEY        = ["store_nbr", "family"]
STORE_TYPE_MAP = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}


def _date_safe_lag(df: pd.DataFrame, col: str, lag: int) -> pd.Series:
    """Compute a lag by merging on (date - lag days).

    This is equivalent to shift(lag) on a perfectly gapless daily series,
    but works correctly even when Dec-25 rows are missing, because it
    looks up the actual past date instead of relying on row position.
    """
    ref = df[KEY + ["date", col]].copy()
    ref = ref.rename(columns={"date": "ref_date", col: f"_lag_src"})
    merged = df[KEY + ["date"]].copy()
    merged["ref_date"] = merged["date"] - pd.Timedelta(days=lag)
    merged = merged.merge(ref, on=KEY + ["ref_date"], how="left")
    return merged["_lag_src"].values


def _date_safe_roll_mean(df: pd.DataFrame, col: str, window: int) -> np.ndarray:
    """Rolling mean over the last `window` calendar days (exclusive of today).

    For each row, we average all rows of the same series whose date is in
    [date - window, date - 1]. Again date-merge based, not position-shift.
    """
    # Build daily sums/counts per series to aggregate efficiently
    agg = df.groupby(KEY + ["date"])[col].mean().reset_index()
    agg.columns = KEY + ["date", "_val"]

    results = np.full(len(df), np.nan)
    # Process series-by-series (manageable: 54 stores x 33 families = 1782)
    for keys, grp in df.groupby(KEY):
        idx = grp.index
        dates = grp["date"].values
        s_agg = agg[(agg["store_nbr"] == keys[0]) & (agg["family"] == keys[1])].copy()
        s_agg = s_agg.sort_values("date").set_index("date")["_val"]
        for i, d in zip(idx, dates):
            lo = d - np.timedelta64(window, "D")
            hi = d - np.timedelta64(1, "D")
            vals = s_agg[(s_agg.index >= lo) & (s_agg.index <= hi)]
            if len(vals):
                results[i] = vals.mean()
    return results


def _date_safe_roll_std(df: pd.DataFrame, col: str, window: int) -> np.ndarray:
    """Rolling std over last `window` calendar days (exclusive of today)."""
    agg = df.groupby(KEY + ["date"])[col].mean().reset_index()
    agg.columns = KEY + ["date", "_val"]

    results = np.full(len(df), np.nan)
    for keys, grp in df.groupby(KEY):
        idx = grp.index
        dates = grp["date"].values
        s_agg = agg[(agg["store_nbr"] == keys[0]) & (agg["family"] == keys[1])].copy()
        s_agg = s_agg.sort_values("date").set_index("date")["_val"]
        for i, d in zip(idx, dates):
            lo = d - np.timedelta64(window, "D")
            hi = d - np.timedelta64(1, "D")
            vals = s_agg[(s_agg.index >= lo) & (s_agg.index <= hi)]
            if len(vals) >= 2:
                results[i] = vals.std()
    return results


# ---------------------------------------------------------------------------
# Fast vectorised lag/rolling using sorted group operations
# (replaces the per-row loops for large data)
# ---------------------------------------------------------------------------
def _fast_lag(df: pd.DataFrame, col: str, lag: int) -> np.ndarray:
    """Vectorised date-safe lag using merge."""
    right = (
        df[KEY + ["date", col]]
        .rename(columns={"date": "_ref", col: "_v"})
    )
    left = df[KEY + ["date"]].copy()
    left["_ref"] = left["date"] - pd.Timedelta(days=lag)
    out = left.merge(right, on=KEY + ["_ref"], how="left")["_v"].values
    return out



def _fast_roll(
    df: pd.DataFrame,
    col: str,
    window: int,
    stat: str = "mean"
) -> np.ndarray:
    """
    Calendar-based rolling statistics.

    Uses the previous `window` calendar days, excluding today.
    Missing dates are handled correctly.
    """

    if stat not in ("mean", "std"):
        raise ValueError(f"Unsupported rolling statistic: {stat}")

    min_periods = 1 if stat == "mean" else 2

    result = pd.Series(
        np.nan,
        index=df.index,
        dtype="float64"
    )

    for _, group in df.groupby(KEY, sort=False):
        group = group.sort_values("date")

        values = pd.Series(
            group[col].to_numpy(),
            index=pd.DatetimeIndex(group["date"])
        )

        rolling = values.rolling(
            window=f"{window}D",
            closed="left",
            min_periods=min_periods
        )

        if stat == "mean":
            calculated = rolling.mean()
        else:
            calculated = rolling.std()

        result.loc[group.index] = calculated.to_numpy()

    return result.to_numpy()



# ---------------------------------------------------------------------------
# Calendar features
# ---------------------------------------------------------------------------
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df["date"]
    df["day_of_week"]      = d.dt.day_of_week.astype("int8")
    df["day_of_month"]     = d.dt.day.astype("int8")
    df["week_of_year"]     = d.dt.isocalendar().week.astype("int8")
    df["month"]            = d.dt.month.astype("int8")
    df["quarter"]          = d.dt.quarter.astype("int8")
    df["year"]             = d.dt.year.astype("int16")
    df["is_weekend"]       = (d.dt.day_of_week >= 5).astype("int8")
    # Days until end of month (0 on last day)
    df["days_to_month_end"] = (
        d + pd.offsets.MonthEnd(0) - d
    ).dt.days.astype("int8")
    # Payday: 15th and last day of month
    df["is_payday"] = (
        (d.dt.day == 15) | (df["days_to_month_end"] == 0)
    ).astype("int8")
    return df


# ---------------------------------------------------------------------------
# Lag features (sales + promo + oil)
# ---------------------------------------------------------------------------
def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    print("  Computing sales lags ...")
    for lag in LAG_DAYS:
        df[f"sales_lag_{lag}"] = _fast_lag(df, "sales", lag).astype("float32")

    print("  Computing promo lags ...")
    df["promo_lag_7"] = _fast_lag(df, "onpromotion", 7).astype("float32")

    print("  Computing oil lags ...")
    df["oil_lag_7"] = _fast_lag(df, "oil_price", 7).astype("float32")

    return df


# ---------------------------------------------------------------------------
# Rolling features
# ---------------------------------------------------------------------------
def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    print("  Computing sales rolling means ...")
    for w in ROLL_WINS:
        df[f"sales_roll_{w}_mean"] = _fast_roll(df, "sales", w, "mean").astype("float32")

    print("  Computing sales rolling std ...")
    df["sales_roll_7_std"] = _fast_roll(df, "sales", 7, "std").astype("float32")

    print("  Computing promo rolling mean ...")
    df["promo_roll_7_mean"] = _fast_roll(df, "onpromotion", 7, "mean").astype("float32")

    print("  Computing oil rolling mean ...")
    df["oil_roll_7_mean"] = _fast_roll(df, "oil_price", 7, "mean").astype("float32")

    return df


# ---------------------------------------------------------------------------
# Encoding (fit on train window, apply everywhere)
# ---------------------------------------------------------------------------
def add_encodings(df: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    # Store type encoding
    df["store_type_enc"] = (
        df["store_type"]
        .map(STORE_TYPE_MAP)
        .astype("int8")
    )

    # Initialize mean encoding
    df["store_family_mean_sales"] = np.nan

    # Extract training data and sort chronologically
    train_rows = df.loc[train_mask, KEY + ["date", "sales"]].copy()
    train_rows = train_rows.sort_values(KEY + ["date"])

    # Historical expanding mean: only previous sales are used
    train_rows["store_family_mean_sales"] = (
        train_rows.groupby(KEY, sort=False)["sales"]
        .transform(
            lambda s: s.shift(1).expanding(min_periods=1).mean()
        )
    )

    # Assign historical means back to training rows
    df.loc[
        train_rows.index, "store_family_mean_sales"
    ] = train_rows["store_family_mean_sales"]

    # Calculate final training mean for validation and test
    series_mean = (
        train_rows.groupby(KEY)["sales"]
        .mean()
    )

    future_mask = ~train_mask

    future_index = pd.MultiIndex.from_frame(
        df.loc[future_mask, KEY]
    )

    df.loc[
        future_mask, "store_family_mean_sales"
    ] = series_mean.reindex(future_index).to_numpy()

    df["store_family_mean_sales"] = (
        df["store_family_mean_sales"].astype("float32")
    )

    # Ordinal encoding for categorical features
    for col, new_col in [
        ("family", "family_enc"),
        ("city", "city_enc"),
        ("state", "state_enc")
    ]:
        cats = df.loc[train_mask, col].astype("category").cat.categories
        mapping = {v: i for i, v in enumerate(cats)}

        df[new_col] = (
            df[col].map(mapping).fillna(-1).astype("int16")
        )

    return df

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_features(
    train_clean: pd.DataFrame,
    test_clean: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Add all features; return (train_features, test_features|None)."""

    # Stack train + test so lags from train spill into test correctly
    if test_clean is not None:
        # test has no 'sales' or qc_ columns — add placeholder sales=NaN
        test_ext = test_clean.copy()
        for c in ["sales"] + config.QC_FLAGS:
            if c not in test_ext.columns:
                test_ext[c] = np.nan
        combined = pd.concat([train_clean, test_ext], ignore_index=True)
        is_test = combined["date"] >= pd.Timestamp(config.VAL_START)
        # Actually test.csv starts 2017-08-16 (after val window); keep them separate
        is_kaggle_test = combined["date"] > pd.Timestamp(config.VAL_END)
    else:
        combined = train_clean.copy()
        is_kaggle_test = pd.Series(False, index=combined.index)

    combined["date"] = pd.to_datetime(combined["date"])
    if combined.duplicated(KEY + ["date"]).any():
        raise ValueError(
            "Duplicate records detected for (store_nbr, family, date)."
        )
    train_mask = combined["date"] <= pd.Timestamp(config.TRAIN_END)

    print("[FE] Calendar features ...")
    combined = add_calendar_features(combined)

    print("[FE] Lag features ...")
    combined = add_lag_features(combined)

    print("[FE] Rolling features ...")
    combined = add_rolling_features(combined)

    print("[FE] Encoding features ...")
    combined = add_encodings(combined, train_mask)

    # Split back
    if test_clean is not None:
        train_out = combined[~is_kaggle_test].reset_index(drop=True)
        test_out  = combined[is_kaggle_test].reset_index(drop=True)
    else:
        train_out = combined.reset_index(drop=True)
        test_out  = None

    return train_out, test_out


def write_feature_report(df: pd.DataFrame) -> None:
    train_mask = df["date"] <= pd.Timestamp(config.TRAIN_END)
    val_mask   = (df["date"] > pd.Timestamp(config.TRAIN_END)) & (df["date"] <= pd.Timestamp(config.VAL_END))

    lag_cols  = [c for c in df.columns if c.startswith("sales_lag_")]
    roll_cols = [c for c in df.columns if "roll" in c]
    cal_cols  = ["day_of_week", "day_of_month", "week_of_year", "month",
                 "quarter", "year", "is_weekend", "days_to_month_end", "is_payday"]
    enc_cols  = ["store_type_enc", "store_family_mean_sales",
                 "family_enc", "city_enc", "state_enc"]
    other_lag = ["promo_lag_7", "oil_lag_7"]
    
    all_feat = list(dict.fromkeys(
        lag_cols + roll_cols + cal_cols + enc_cols + other_lag
    ))

    # Count NaNs (expected in early rows before lag data exists)
    nan_counts = df[all_feat].isna().sum()

    lines = [
        "# StockWise AI — Feature Engineering Report (auto-generated by src/feature_engineering.py)",
        "",
        "Re-run `python src/feature_engineering.py` to refresh.",
        "",
        f"## Dataset size",
        f"- Total rows : {len(df):,}",
        f"- Train rows (≤ {config.TRAIN_END}) : {train_mask.sum():,}",
        f"- Val rows ({config.VAL_START} → {config.VAL_END}) : {val_mask.sum():,}",
        f"- Columns : {df.shape[1]}",
        "",
        "## Feature groups",
        "",
        f"| Group | Features | Count |",
        f"|---|---|---|",
        f"| Calendar | {', '.join(f'`{c}`' for c in cal_cols)} | {len(cal_cols)} |",
        f"| Sales lags | {', '.join(f'`{c}`' for c in lag_cols)} | {len(lag_cols)} |",
        f"| Rolling stats | {', '.join(f'`{c}`' for c in roll_cols)} | {len(roll_cols)} |",
        f"| Promo/Oil lags | {', '.join(f'`{c}`' for c in other_lag)} | {len(other_lag)} |",
        f"| Encodings | {', '.join(f'`{c}`' for c in enc_cols)} | {len(enc_cols)} |",
        "",
        "## NaN counts in feature columns (expected for early-history rows)",
        "",
        "| Feature | NaN rows |",
        "|---|---|",
    ]
    for col in all_feat:
        n = int(nan_counts[col])
        if n > 0:
            lines.append(f"| `{col}` | {n:,} |")
    lines += [
        "",
        "> [!NOTE]",
        "> NaNs in lag/rolling columns are expected for the first N rows of each series",
        "> (before N days of history exist). The models handle these via imputation.",
        "",
        "## QC rows excluded from training",
        f"- `qc_zero_series = 1`    : {int(df['qc_zero_series'].sum()):,} rows (whole series never sold)",
        f"- `qc_before_first_sale = 1` : {int(df['qc_before_first_sale'].sum()):,} rows (store/product not opened yet)",
    ]

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.FEATURE_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved: {config.FEATURE_REPORT_PATH}")


def main(argv: list[str] | None = None) -> int:
    setup_console()
    ap = argparse.ArgumentParser(description="StockWise AI: build features.")
    ap.add_argument("--no-test", action="store_true", help="skip test_clean.parquet enrichment")
    args = ap.parse_args(argv)

    if not config.TRAIN_CLEAN_PATH.exists():
        print("train_clean.parquet not found. Run first:  python src/prepare_data.py")
        return 1

    try:
        import pyarrow  # noqa: F401
    except ImportError:
        print("pyarrow is missing. Run:  pip install pyarrow")
        return 1

    t0 = time.time()
    print(f"[FE] Loading {config.TRAIN_CLEAN_PATH.name} ...")
    train_clean = pd.read_parquet(config.TRAIN_CLEAN_PATH)

    test_clean = None
    if not args.no_test and config.TEST_CLEAN_PATH.exists():
        print(f"[FE] Loading {config.TEST_CLEAN_PATH.name} ...")
        test_clean = pd.read_parquet(config.TEST_CLEAN_PATH)

    print(f"[FE] Train rows: {len(train_clean):,}  |  date range: "
          f"{train_clean['date'].min().date()} → {train_clean['date'].max().date()}")

    train_feat, test_feat = build_features(train_clean, test_clean)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[FE] Saving train_features.parquet ({len(train_feat):,} rows, {train_feat.shape[1]} cols) ...")
    train_feat.to_parquet(config.FEATURES_TRAIN_PATH, index=False)

    if test_feat is not None:
        print(f"[FE] Saving test_features.parquet ({len(test_feat):,} rows) ...")
        test_feat.to_parquet(config.FEATURES_TEST_PATH, index=False)

    print("\n[FE] Writing feature report ...")
    write_feature_report(train_feat)

    elapsed = time.time() - t0
    print(f"\n✓ Feature engineering done in {elapsed:.1f}s")
    print(f"  → {config.FEATURES_TRAIN_PATH}")
    if test_feat is not None:
        print(f"  → {config.FEATURES_TEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


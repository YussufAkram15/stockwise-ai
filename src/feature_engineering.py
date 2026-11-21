"""STEP 7: Feature engineering for StockWise AI.

Run from the project root (after prepare_data.py):
    python src/feature_engineering.py
    python src/feature_engineering.py --no-test

Forecast horizon: 16 days.
Sales-derived features use sales_capped (when supplied by prepare_data.py) and
are shifted so a prediction for day t only sees history available through t-16.
Calendar and promotion features are used as known-in-advance information.
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

KEY = ["store_nbr", "family"]
HORIZON_DAYS = 16
LAG_DAYS = [16, 21, 28, 35, 42]
SAME_WEEKDAY_LAGS = [21, 28, 35, 42]
ROLL_WINS = [7, 28]
HOLIDAY_CAP = 30
STORE_TYPE_MAP = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
HISTORY_COL = "sales_capped"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _validate_input(df: pd.DataFrame, name: str) -> None:
    required = set(KEY + ["date", "sales", "onpromotion", "oil_price",
                          "store_type", "family", "city", "state", "is_holiday"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")

    if df[KEY + ["date"]].isna().any().any():
        raise ValueError(f"{name} contains null values in store_nbr/family/date.")

    duplicated = df.duplicated(KEY + ["date"])
    if duplicated.any():
        sample = df.loc[duplicated, KEY + ["date"]].head(5).to_dict("records")
        raise ValueError(
            f"{name} contains duplicate (store_nbr, family, date) rows. "
            f"Examples: {sample}"
        )


# ---------------------------------------------------------------------------
# Date-aware building blocks
# ---------------------------------------------------------------------------
def _fast_lag(
    df: pd.DataFrame,
    col: str,
    lag: int,
    tolerance_days: int = 1,
) -> np.ndarray:
    """Get a value near (date - lag) within the same store/family series.

    Exact date matches are preferred. If the expected date is absent, the
    latest earlier date is accepted only when it is at most tolerance_days
    earlier. This handles isolated one-day calendar gaps (for example Dec 25)
    without ever taking a value from the future.

    The returned array is aligned to the original row order of df.
    """
    if lag < 0:
        raise ValueError("lag must be >= 0")
    if tolerance_days < 0:
        raise ValueError("tolerance_days must be >= 0")
    if col not in df.columns:
        raise KeyError(f"Cannot calculate lag: column '{col}' does not exist.")

    # Keep a stable positional key because merge_asof sorts by the date key.
    left = df[KEY + ["date"]].copy()
    left["_pos"] = np.arange(len(left), dtype=np.int64)
    left["_ref"] = left["date"] - pd.Timedelta(days=lag)

    right = df[KEY + ["date", col]].rename(
        columns={"date": "_ref", col: "_value"}
    )

    # merge_asof requires the 'on' key to be globally sorted, not merely
    # sorted within each store/family group.
    left = left.sort_values(["_ref", *KEY], kind="mergesort")
    right = right.sort_values(["_ref", *KEY], kind="mergesort")

    merged = pd.merge_asof(
        left,
        right,
        on="_ref",
        by=KEY,
        direction="backward",
        tolerance=pd.Timedelta(days=tolerance_days),
        allow_exact_matches=True,
    )
    return (
        merged.sort_values("_pos", kind="mergesort")["_value"]
        .to_numpy()
    )


def _fast_roll(
    df: pd.DataFrame,
    col: str,
    window: int,
    closed: str,
) -> np.ndarray:
    """Calendar-day rolling mean per series, returned in original row order.

    closed='right': window ending today, including today.
    closed='left' : window before today, excluding today.
    """
    if col not in df.columns:
        raise KeyError(f"Cannot calculate rolling feature: '{col}' is missing.")
    if window <= 0:
        raise ValueError("window must be positive.")
    if closed not in {"left", "right", "both", "neither"}:
        raise ValueError("closed must be left, right, both, or neither.")

    result = pd.Series(np.nan, index=df.index, dtype="float64")
    for _, group in df.groupby(KEY, sort=False, observed=True):
        group = group.sort_values("date", kind="mergesort")
        values = pd.Series(
            group[col].to_numpy(dtype="float64"),
            index=pd.DatetimeIndex(group["date"]),
        )
        rolled = values.rolling(
            window=f"{window}D", closed=closed, min_periods=1
        ).mean()
        result.loc[group.index] = rolled.to_numpy()

    return result.to_numpy()


# ---------------------------------------------------------------------------
# Calendar features (known in advance)
# ---------------------------------------------------------------------------
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df["date"]
    df["day_of_week"] = d.dt.day_of_week.astype("int8")
    df["day_of_month"] = d.dt.day.astype("int8")
    df["week_of_year"] = d.dt.isocalendar().week.astype("int8")
    df["month"] = d.dt.month.astype("int8")
    df["quarter"] = d.dt.quarter.astype("int8")
    df["year"] = d.dt.year.astype("int16")
    df["is_weekend"] = (d.dt.day_of_week >= 5).astype("int8")
    month_end = d + pd.offsets.MonthEnd(0)
    df["days_to_month_end"] = (month_end - d).dt.days.astype("int8")
    df["is_payday"] = (
        (d.dt.day == 15) | (df["days_to_month_end"] == 0)
    ).astype("int8")
    return df


def add_holiday_proximity(df: pd.DataFrame) -> pd.DataFrame:
    """Days to/from the nearest holiday for each store, capped at 30."""
    # Aggregate rather than arbitrarily keeping one family row. Holiday status
    # should be the same across families, and max makes the lookup robust if
    # the source contains inconsistent duplicate calendar flags.
    cal = (
        df[["store_nbr", "date", "is_holiday"]]
        .assign(is_holiday=lambda x: pd.to_numeric(
            x["is_holiday"], errors="coerce"
        ).fillna(0))
        .groupby(["store_nbr", "date"], as_index=False, sort=False)["is_holiday"]
        .max()
        .sort_values(["store_nbr", "date"], kind="mergesort")
        .reset_index(drop=True)
    )

    holiday_date = cal["date"].where(cal["is_holiday"] > 0)
    prev_holiday = holiday_date.groupby(cal["store_nbr"], sort=False).ffill()
    next_holiday = holiday_date.groupby(cal["store_nbr"], sort=False).bfill()

    cal["days_since_last_holiday"] = (cal["date"] - prev_holiday).dt.days
    cal["days_to_next_holiday"] = (next_holiday - cal["date"]).dt.days

    lookup = cal.set_index(["store_nbr", "date"])
    row_key = pd.MultiIndex.from_frame(df[["store_nbr", "date"]])

    for col in ("days_since_last_holiday", "days_to_next_holiday"):
        values = lookup[col].reindex(row_key).to_numpy(dtype="float64")
        df[col] = (
            pd.Series(values, index=df.index)
            .fillna(HOLIDAY_CAP)
            .clip(lower=0, upper=HOLIDAY_CAP)
            .astype("int8")
        )
    return df


# ---------------------------------------------------------------------------
# Sales / promotion / oil history
# ---------------------------------------------------------------------------
def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    print("  Computing sales lags ...")
    for lag in LAG_DAYS:
        df[f"sales_lag_{lag}"] = _fast_lag(
            df, HISTORY_COL, lag, tolerance_days=1
        ).astype("float32")

    df["sales_same_weekday_mean"] = (
        df[[f"sales_lag_{lag}" for lag in SAME_WEEKDAY_LAGS]]
        .mean(axis=1)
        .astype("float32")
    )

    print("  Computing promotion and oil lags ...")
    # Promotion/oil values are date-level known observations. The same
    # one-day tolerance protects against an isolated missing calendar date.
    df["promo_lag_7"] = _fast_lag(
        df, "onpromotion", 7, tolerance_days=1
    ).astype("float32")

    for lag in (16, 21, 28):
        df[f"oil_lag_{lag}"] = _fast_lag(
            df, "oil_price", lag, tolerance_days=1
        ).astype("float32")
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    print("  Computing rolling means ...")
    for window in ROLL_WINS:
        # First calculate a rolling mean ending on each historical date,
        # then retrieve that value from t-16 for the target date t.
        rolling_col = f"_sales_roll_{window}"
        df[rolling_col] = _fast_roll(
            df, HISTORY_COL, window, closed="right"
        )
        df[f"sales_roll_{window}_mean"] = _fast_lag(
            df, rolling_col, HORIZON_DAYS, tolerance_days=1
        ).astype("float32")

    df.drop(columns=[f"_sales_roll_{w}" for w in ROLL_WINS], inplace=True)

    # Promotions are planned in advance; the seven calendar days before the
    # target date can therefore be used without the sales-history embargo.
    df["promo_roll_7_mean"] = _fast_roll(
        df, "onpromotion", 7, closed="left"
    ).astype("float32")
    return df


def add_series_level_and_codes(
    df: pd.DataFrame,
    train_mask: pd.Series,
) -> pd.DataFrame:
    print("  Computing series level and label codes ...")

    # Expanding average through each historical date, shifted back by 16 days.
    d = (
        df[KEY + ["date", HISTORY_COL]]
        .copy()
        .sort_values(KEY + ["date"], kind="mergesort")
    )
    groups = [d["store_nbr"], d["family"]]
    valid_history = d[HISTORY_COL].notna().astype("int64")
    cumulative_total = (
        d[HISTORY_COL].fillna(0).groupby(groups, sort=False).cumsum()
    )
    cumulative_count = valid_history.groupby(groups, sort=False).cumsum()
    d["_cum_mean"] = cumulative_total / cumulative_count.where(
        cumulative_count > 0
    )

    # d has a fresh positional index after the sort, so assign the result by
    # its original row labels before aligning back to df.
    level_values = _fast_lag(
        d, "_cum_mean", HORIZON_DAYS, tolerance_days=1
    )
    level = pd.Series(level_values, index=d.index)
    df["store_family_mean_sales"] = (
        level.reindex(df.index).astype("float32")
    )

    # Fit category mappings on training-period rows only.
    df["store_type_enc"] = (
        df["store_type"].map(STORE_TYPE_MAP).fillna(-1).astype("int8")
    )
    for source_col, output_col in (
        ("family", "family_enc"),
        ("city", "city_enc"),
        ("state", "state_enc"),
    ):
        categories = (
            df.loc[train_mask, source_col]
            .dropna()
            .astype("category")
            .cat.categories
        )
        mapping = {value: code for code, value in enumerate(categories)}
        df[output_col] = (
            df[source_col].map(mapping).fillna(-1).astype("int16")
        )
    return df


# ---------------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------------
def build_features(
    train_clean: pd.DataFrame,
    test_clean: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Build features for train/validation and optional Kaggle test rows."""
    train = train_clean.copy()
    train["date"] = pd.to_datetime(train["date"], errors="raise")
    _validate_input(train, "train_clean")

    if HISTORY_COL not in train.columns:
        # Backward compatibility for older prepared datasets. If the column
        # exists, preserve its NaNs: filling them from raw sales could undo
        # intentional QC handling performed by prepare_data.py.
        train[HISTORY_COL] = train["sales"]

    if test_clean is not None:
        test = test_clean.copy()
        test["date"] = pd.to_datetime(test["date"], errors="raise")

        # Kaggle/test data may not contain the target column at all.
        # Add it before validation so future rows remain explicitly unknown.
        for col in ["sales", HISTORY_COL] + list(config.QC_FLAGS):
            if col not in test.columns:
                test[col] = np.nan

        _validate_input(test, "test_clean")
        combined = pd.concat([train, test], ignore_index=True, sort=False)
    else:
        combined = train.reset_index(drop=True)

    combined["date"] = pd.to_datetime(combined["date"], errors="raise")
    _validate_input(combined, "combined train/test data")

    # Do not fill null sales_capped values from raw sales when the column
    # already exists. NaN may be intentional (missing or excluded history).
    if HISTORY_COL not in combined.columns:
        combined[HISTORY_COL] = combined["sales"]

    train_mask = combined["date"] <= pd.Timestamp(config.TRAIN_END)
    test_mask = (
        combined["date"] > pd.Timestamp(config.TEST_END)
        if test_clean is not None
        else pd.Series(False, index=combined.index)
    )

    print("[FE] Calendar clues ...")
    combined = add_calendar_features(combined)
    combined = add_holiday_proximity(combined)

    print("[FE] Lag clues ...")
    combined = add_lag_features(combined)

    print("[FE] Rolling clues ...")
    combined = add_rolling_features(combined)

    print("[FE] Series level and label codes ...")
    combined = add_series_level_and_codes(combined, train_mask)

    if test_clean is not None:
        train_features = combined.loc[~test_mask].reset_index(drop=True)
        test_features = combined.loc[test_mask].reset_index(drop=True)
        return train_features, test_features

    return combined.reset_index(drop=True), None


def new_feature_columns(df: pd.DataFrame) -> list[str]:
    groups = (
        [
            "day_of_week", "day_of_month", "week_of_year", "month",
            "quarter", "year", "is_weekend", "days_to_month_end",
            "is_payday", "days_to_next_holiday", "days_since_last_holiday",
        ],
        [f"sales_lag_{lag}" for lag in LAG_DAYS]
        + ["sales_same_weekday_mean"]
        + [f"sales_roll_{window}_mean" for window in ROLL_WINS],
        [
            "promo_lag_7", "promo_roll_7_mean",
            "oil_lag_16", "oil_lag_21", "oil_lag_28",
        ],
        [
            "store_family_mean_sales", "store_type_enc", "family_enc",
            "city_enc", "state_enc",
        ],
    )
    return [column for group in groups for column in group if column in df.columns]


def write_feature_report(df: pd.DataFrame) -> None:
    train_mask = df["date"] <= pd.Timestamp(config.TRAIN_END)
    val_mask = (
        (df["date"] > pd.Timestamp(config.TRAIN_END))
        & (df["date"] <= pd.Timestamp(config.VAL_END))
    )
    columns = new_feature_columns(df)
    nan_counts = df[columns].isna().sum()

    calendar_cols = columns[:11]
    sales_cols = [
        c for c in columns
        if c.startswith("sales_") and c != "store_family_mean_sales"
    ]
    promo_oil_cols = [
        c for c in columns if c.startswith(("promo_", "oil_"))
    ]

    lines = [
        "# StockWise AI - Feature report",
        "",
        "Forecast horizon: 16 days. Sales-history features are shifted by 16 days.",
        "Run `python src/leakage_check.py` separately to test for leakage.",
        "",
        "## Size",
        f"- Rows: {len(df):,}; columns: {df.shape[1]}",
        f"- Training-period rows (<= {config.TRAIN_END}): {int(train_mask.sum()):,}",
        f"- Rows between {config.TRAIN_END} and {config.VAL_END}: {int(val_mask.sum()):,}",
        "",
        "## Feature groups",
        "",
        "| Group | Features |",
        "|---|---|",
        "| Calendar | " + ", ".join(f"`{c}`" for c in calendar_cols) + " |",
        "| Sales history | " + ", ".join(f"`{c}`" for c in sales_cols) + " |",
        "| Promotion / oil | " + ", ".join(f"`{c}`" for c in promo_oil_cols) + " |",
        "| Level / labels | `store_family_mean_sales`, `store_type_enc`, `family_enc`, `city_enc`, `state_enc` |",
        "",
        "## Missing values",
        "",
        "| Feature | Missing rows |",
        "|---|---:|",
    ]
    lines.extend(
        f"| `{column}` | {int(count):,} |"
        for column, count in nan_counts.items()
        if count > 0
    )

    lines.extend(["", "## QC flags in this dataset"])
    for flag, description in (
        ("qc_zero_series", "series never sold"),
        ("qc_before_first_sale", "before first sale"),
        ("qc_sales_suspect", "suspected sales data errors"),
    ):
        if flag in df.columns:
            count = int(pd.to_numeric(df[flag], errors="coerce").fillna(0).sum())
            lines.append(f"- `{flag} = 1`: {count:,} rows ({description})")

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.FEATURE_REPORT_PATH.write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"Saved: {config.FEATURE_REPORT_PATH}")


def main(argv: list[str] | None = None) -> int:
    setup_console()
    parser = argparse.ArgumentParser(
        description="StockWise AI: build model-ready features."
    )
    parser.add_argument(
        "--no-test", action="store_true",
        help="skip test_clean.parquet",
    )
    args = parser.parse_args(argv)

    if not config.TRAIN_CLEAN_PATH.exists():
        print("train_clean.parquet not found. Run: python src/prepare_data.py")
        return 1

    try:
        import pyarrow  # noqa: F401
    except ImportError:
        print("pyarrow is missing. Install it with: pip install pyarrow")
        return 1

    start = time.time()
    print(f"[FE] Loading {config.TRAIN_CLEAN_PATH.name} ...")
    train_clean = pd.read_parquet(config.TRAIN_CLEAN_PATH)

    test_clean = None
    if not args.no_test and config.TEST_CLEAN_PATH.exists():
        print(f"[FE] Loading {config.TEST_CLEAN_PATH.name} ...")
        test_clean = pd.read_parquet(config.TEST_CLEAN_PATH)

    print(
        f"[FE] Train rows: {len(train_clean):,} | "
        f"{pd.to_datetime(train_clean['date']).min().date()} -> "
        f"{pd.to_datetime(train_clean['date']).max().date()}"
    )

    train_features, test_features = build_features(train_clean, test_clean)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"\n[FE] Saving train_features.parquet "
        f"({len(train_features):,} rows, {train_features.shape[1]} columns) ..."
    )
    train_features.to_parquet(config.FEATURES_TRAIN_PATH, index=False)

    if test_features is not None:
        print(f"[FE] Saving test_features.parquet ({len(test_features):,} rows) ...")
        test_features.to_parquet(config.FEATURES_TEST_PATH, index=False)

    write_feature_report(train_features)
    print(f"\nFeature engineering completed in {time.time() - start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

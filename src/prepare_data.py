"""STEPS 2-6 (Zenab): merge + clean the raw tables and save the cleaned files.

Run from the project root:
    python src/prepare_data.py              # clean, save, then run the validation checks
    python src/prepare_data.py --csv        # also write data/processed/train_clean.csv
    python src/prepare_data.py --skip-validation

What it does (every decision is also written to reports/cleaning_report.md):
  2. merge stores.csv and oil.csv into the sales rows
  3. fill missing oil prices (carry the last known price forward)
  4. sort out holidays (transferred holidays are normal days; events are separate)
  5. flag zero-sales series, closed stores and outliers (flagged, NEVER deleted)
  6. save data/processed/train_clean.parquet (+ test_clean.parquet)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import config
from data_io import load_raw, missing_required_files, setup_console


# ----------------------------------------------------------------------------
# Small checks on the raw tables
# ----------------------------------------------------------------------------
def validate_sales_table(df: pd.DataFrame, name: str, has_sales: bool) -> None:
    required = ["id", "date", "store_nbr", "family", "onpromotion"] + (["sales"] if has_sales else [])
    absent = [c for c in required if c not in df.columns]
    if absent:
        raise ValueError(f"{name}: missing columns {absent}")
    nulls = df[required].isna().sum()
    if nulls.any():
        raise ValueError(f"{name}: missing values found: {nulls[nulls > 0].to_dict()}")
    if df["id"].duplicated().any():
        raise ValueError(f"{name}: duplicate ids")
    if df.duplicated(["date", "store_nbr", "family"]).any():
        raise ValueError(f"{name}: duplicate (date, store_nbr, family) rows")
    if has_sales and (df["sales"] < 0).any():
        raise ValueError(f"{name}: negative sales")
    if (df["onpromotion"] < 0).any():
        raise ValueError(f"{name}: negative onpromotion")


def downcast_sales_table(df: pd.DataFrame, has_sales: bool) -> pd.DataFrame:
    """Smaller numbers = smaller file and less memory. Values are not changed."""
    df = df.copy()
    df["id"] = df["id"].astype("int32")
    df["store_nbr"] = df["store_nbr"].astype("int16")
    df["onpromotion"] = df["onpromotion"].astype("int32")
    if has_sales:
        df["sales"] = df["sales"].astype("float64")
    return df


# ----------------------------------------------------------------------------
# Step 2: stores
# ----------------------------------------------------------------------------
def clean_stores(stores: pd.DataFrame) -> pd.DataFrame:
    needed = ["store_nbr", "city", "state", "type", "cluster"]
    absent = [c for c in needed if c not in stores.columns]
    if absent:
        raise ValueError(f"stores.csv: missing columns {absent}")
    if stores[needed].isna().any().any():
        raise ValueError("stores.csv: missing values")
    if stores["store_nbr"].duplicated().any():
        raise ValueError("stores.csv: duplicate store_nbr")
    out = stores[needed].rename(columns={"type": "store_type"}).copy()
    out["store_nbr"] = out["store_nbr"].astype("int16")
    out["cluster"] = out["cluster"].astype("int16")
    return out


# ----------------------------------------------------------------------------
# Step 3: oil  (one price for EVERY calendar day, no blanks)
# ----------------------------------------------------------------------------
def build_oil_daily(oil: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    if oil["date"].duplicated().any():
        raise ValueError("oil.csv: duplicate dates")
    if (oil["dcoilwtico"].dropna() <= 0).any():
        raise ValueError("oil.csv: zero or negative price")

    # Build the calendar from the oil file's own span too, so that the last known
    # price BEFORE `start` can still be carried into the first days.
    span_start = min(start, oil["date"].min())
    span_end = max(end, oil["date"].max())
    daily = pd.DataFrame({"date": pd.date_range(span_start, span_end, freq="D")})
    daily = daily.merge(oil[["date", "dcoilwtico"]], on="date", how="left", validate="one_to_one")

    daily["oil_is_filled"] = daily["dcoilwtico"].isna().astype("int8")
    daily["oil_price"] = daily["dcoilwtico"].ffill()          # carry LAST KNOWN price forward (no future info)
    n_leading = int(daily["oil_price"].isna().sum())           # days before the very first real price
    daily["oil_price"] = daily["oil_price"].bfill()            # only these leading days use the first real price
    if daily["oil_price"].isna().any():
        raise ValueError("oil.csv has no price at all")

    first_real = daily.loc[daily["dcoilwtico"].notna()].iloc[0]
    daily = daily[(daily["date"] >= start) & (daily["date"] <= end)].reset_index(drop=True)
    info = {
        "n_days": len(daily),
        "n_real": int(daily["dcoilwtico"].notna().sum()),
        "n_filled": int(daily["oil_is_filled"].sum()),
        "n_leading_backfilled": n_leading,
        "first_real_date": first_real["date"].date(),
        "first_real_price": float(first_real["dcoilwtico"]),
        "price_min": float(daily["oil_price"].min()),
        "price_max": float(daily["oil_price"].max()),
    }
    return daily[["date", "dcoilwtico", "oil_price", "oil_is_filled"]], info


# ----------------------------------------------------------------------------
# Step 4: holidays
# ----------------------------------------------------------------------------
def classify_holidays(hol: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Give every holiday row ONE category:
        transferred_away : transferred == True -> the holiday did NOT happen on this date (normal day)
        holiday          : a real day off (Holiday / Transfer / Additional / Bridge, not transferred)
        event            : Black Friday, Mother's Day, earthquake ... (not a day off)
        work_day         : a normal working day on a weekend
    """
    hol = hol.drop_duplicates().copy()
    bad_types = sorted(set(hol["type"]) - set(config.KNOWN_HOLIDAY_TYPES))
    if bad_types:
        raise ValueError(f"holidays_events.csv: unknown type values {bad_types}")
    bad_loc = sorted(set(hol["locale"]) - set(config.KNOWN_LOCALES))
    if bad_loc:
        raise ValueError(f"holidays_events.csv: unknown locale values {bad_loc}")

    hol["category"] = np.select(
        [
            hol["transferred"].to_numpy(),
            hol["type"].isin(config.REAL_HOLIDAY_TYPES).to_numpy(),
            hol["type"].eq(config.EVENT_TYPE).to_numpy(),
            hol["type"].eq(config.WORK_DAY_TYPE).to_numpy(),
        ],
        ["transferred_away", "holiday", "event", "work_day"],
        default="unknown",
    )
    if (hol["category"] == "unknown").any():
        raise ValueError("holidays_events.csv: a row could not be classified")
    hol = hol.sort_values(["date", "locale", "locale_name"]).reset_index(drop=True)
    info = {
        "n_rows": len(hol),
        "by_category": hol["category"].value_counts().to_dict(),
        "transferred_rows": hol.loc[hol["category"] == "transferred_away",
                                    ["date", "locale", "locale_name", "description"]],
    }
    return hol, info


def _scope_hits(cal: pd.DataFrame, rows: pd.DataFrame, scope: str) -> np.ndarray:
    """1 for every (date, store) hit by the holiday `rows` of ONE scope.
    National -> every store; Regional -> stores of that state; Local -> stores of that city."""
    if rows.empty:
        return np.zeros(len(cal), dtype="int8")
    if scope == "National":
        return cal["date"].isin(rows["date"].unique()).to_numpy().astype("int8")
    key_col = "state" if scope == "Regional" else "city"
    keys = rows[["date", "locale_name"]].drop_duplicates().rename(columns={"locale_name": key_col})
    keys["_hit"] = 1
    merged = cal[["date", key_col]].merge(keys, on=["date", key_col], how="left", validate="many_to_one")
    if len(merged) != len(cal):
        raise RuntimeError("calendar merge changed the number of rows")
    return merged["_hit"].fillna(0).to_numpy().astype("int8")


def _applies(cal: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    """1 if ANY row (national, regional or local) applies to that store on that date."""
    hit = np.zeros(len(cal), dtype="int8")
    for scope in config.KNOWN_LOCALES:
        hit = hit | _scope_hits(cal, rows[rows["locale"] == scope], scope)
    return hit


def build_calendar(dates: pd.Series, stores: pd.DataFrame, hol: pd.DataFrame) -> pd.DataFrame:
    """One small table with one row per (date, store) and all holiday/event flags.
    (~90k rows instead of 3M, then merged onto the sales rows.)"""
    days = pd.DataFrame({"date": pd.Series(dates).drop_duplicates().sort_values().to_numpy()})
    cal = days.merge(stores[["store_nbr", "city", "state"]], how="cross")

    real = hol[hol["category"] == "holiday"]
    cal["is_holiday_national"] = _scope_hits(cal, real[real["locale"] == "National"], "National")
    cal["is_holiday_regional"] = _scope_hits(cal, real[real["locale"] == "Regional"], "Regional")
    cal["is_holiday_local"] = _scope_hits(cal, real[real["locale"] == "Local"], "Local")
    cal["is_holiday"] = (cal[["is_holiday_national", "is_holiday_regional", "is_holiday_local"]]
                         .max(axis=1).astype("int8"))

    events = hol[hol["category"] == "event"]
    quake = events[events["description"].str.contains("terremoto", case=False, na=False)]
    cal["is_event"] = _applies(cal, events)
    cal["is_earthquake"] = _applies(cal, quake)
    cal["is_work_day"] = _applies(cal, hol[hol["category"] == "work_day"])
    return cal[["date", "store_nbr"] + config.CALENDAR_FLAGS]


# ----------------------------------------------------------------------------
# Merge everything onto the sales rows
# ----------------------------------------------------------------------------
def enrich(df: pd.DataFrame, stores: pd.DataFrame, oil_daily: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    out = df.merge(stores, on="store_nbr", how="left", validate="many_to_one")
    out = out.merge(oil_daily[["date", "oil_price", "oil_is_filled"]], on="date", how="left", validate="many_to_one")
    out = out.merge(calendar, on=["date", "store_nbr"], how="left", validate="many_to_one")
    if len(out) != n:
        raise RuntimeError("a merge changed the number of rows")
    new_cols = ["city", "state", "store_type", "cluster", "oil_price", "oil_is_filled"] + config.CALENDAR_FLAGS
    nulls = out[new_cols].isna().sum()
    if nulls.any():
        raise RuntimeError(f"merge left missing values: {nulls[nulls > 0].to_dict()}")
    return out


# ----------------------------------------------------------------------------
# Step 5: zero series, closed stores, outliers  (FLAGGED, never deleted)
# ----------------------------------------------------------------------------
def add_qc_flags(train: pd.DataFrame, cutoff: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = ["store_nbr", "family"]
    cutoff_ts = pd.Timestamp(cutoff)
    train = train.sort_values(key + ["date"], kind="mergesort").reset_index(drop=True)

    # Series statistics learned ONLY from the training window (date <= cutoff).
    window = train.loc[train["date"] <= cutoff_ts, key + ["date", "sales"]]
    window = window.assign(
        sale_date=window["date"].where(window["sales"] > 0),
        is_zero=(window["sales"] == 0).astype("int8"),
    )
    summary = window.groupby(key).agg(
        n_days=("sales", "size"),
        n_zero_days=("is_zero", "sum"),
        total_sales=("sales", "sum"),
        first_sale_date=("sale_date", "min"),
        last_sale_date=("sale_date", "max"),
    )
    nonzero = window.loc[window["sales"] > 0]
    quart = nonzero.groupby(key)["sales"].quantile([0.25, 0.75]).unstack()
    quart.columns = ["q1", "q3"]
    summary = summary.join(quart)
    summary["n_nonzero_days"] = summary["n_days"] - summary["n_zero_days"]
    summary["zero_share"] = summary["n_zero_days"] / summary["n_days"]
    summary["qc_zero_series"] = (summary["total_sales"] == 0).astype("int8")
    limit = summary["q3"] + config.OUTLIER_IQR_MULT * (summary["q3"] - summary["q1"])
    summary["outlier_limit"] = limit.where(summary["n_nonzero_days"] >= config.OUTLIER_MIN_NONZERO_DAYS)
    summary = summary.reset_index()

    helper = summary[key + ["first_sale_date", "qc_zero_series", "outlier_limit"]]
    train = train.merge(helper, on=key, how="left", validate="many_to_one")
    train["qc_zero_series"] = train["qc_zero_series"].fillna(0).astype("int8")

    # before the first sale of the series (only meaningful inside the window)
    train["qc_before_first_sale"] = (
        (train["date"] <= cutoff_ts)
        & (train["first_sale_date"].isna() | (train["date"] < train["first_sale_date"]))
    ).astype("int8")

    # the whole store sold nothing that day (closed, or not opened yet)
    store_day_total = train.groupby(["store_nbr", "date"])["sales"].transform("sum")
    train["qc_store_closed_day"] = (store_day_total == 0).astype("int8")

    # unusually high sale for its own series (NaN limit -> False -> 0)
    train["qc_sales_outlier"] = (train["sales"] > train["outlier_limit"]).astype("int8")

    return train.drop(columns=["first_sale_date", "outlier_limit"]), summary


# ----------------------------------------------------------------------------
# Cleaning report (every decision written down, with real numbers)
# ----------------------------------------------------------------------------
def write_cleaning_report(ctx: dict) -> None:
    tr, oil, hol, qc = ctx["train_clean"], ctx["oil_info"], ctx["hol_info"], ctx["qc"]
    n = len(tr)
    pct = lambda s: f"{100 * float(s.mean()):.2f}%"
    cat = hol["by_category"]

    L: list[str] = []
    add = L.append
    add("# StockWise AI - Cleaning report (auto-generated by src/prepare_data.py)")
    add("")
    add("Re-run `python src/prepare_data.py` and this file is rewritten with the new numbers.")
    add("")
    add("## 1. Result")
    add(f"- `data/processed/train_clean.parquet`: **{n:,} rows x {tr.shape[1]} columns**, "
        f"{tr['date'].min().date()} -> {tr['date'].max().date()}, "
        f"{tr['store_nbr'].nunique()} stores x {tr['family'].nunique()} families.")
    add("- **No row was added or removed and `sales` was not changed.** Cleaning only adds columns.")
    add("- Sorted by `store_nbr`, `family`, `date`: each series is one block, oldest day first "
        "(safe for `groupby(...).shift()` and rolling windows).")
    if ctx["test_clean"] is not None:
        add(f"- `data/processed/test_clean.parquet`: {len(ctx['test_clean']):,} rows (Kaggle test period), "
            "same enrichment, original `id` order, no `sales` and no `qc_` columns.")
    add("")
    add("## 2. Decisions")
    add("### Stores")
    add("- Left join on `store_nbr`. `type` renamed to `store_type` (it would clash with the holiday `type`).")
    add("### Oil")
    add(f"- oil.csv only has business days, and some prices are blank. We built one row per calendar day "
        f"({oil['n_days']:,} days, train + test period, Dec 25 included) and **carried the last known price forward** "
        "(uses only the past, so no leak).")
    add(f"- {oil['n_real']:,} days have a real price, **{oil['n_filled']:,} days were filled** "
        f"(flag `oil_is_filled` = 1).")
    add(f"- {oil['n_leading_backfilled']} day(s) at the very start have no earlier price; they take the first real price "
        f"({oil['first_real_price']:.2f} on {oil['first_real_date']}).")
    add(f"- Price range after filling: {oil['price_min']:.2f} -> {oil['price_max']:.2f}.")
    add("### Holidays")
    add(f"- {hol['n_rows']} rows: " + ", ".join(f"{k} = {v}" for k, v in sorted(cat.items())) + ".")
    add("- `transferred = True` means the holiday was moved: **that date is a normal day** "
        "(category `transferred_away`). The day it moved TO appears as type `Transfer` and counts as a real holiday.")
    add(f"- Real holiday = type in {list(config.REAL_HOLIDAY_TYPES)} and not transferred. "
        "`Event` rows (Black Friday, Mother's Day, earthquake...) and `Work Day` rows are NOT holidays; they have their own flags.")
    add("- Scope: National -> all stores; Regional -> stores whose `state` = locale_name; Local -> stores whose `city` = locale_name.")
    if ctx["unmatched"]:
        add("- Holiday places with no store (ignored automatically): " + ", ".join(ctx["unmatched"]) + ".")
    add(f"- Share of rows: `is_holiday` {pct(tr['is_holiday'])}, `is_event` {pct(tr['is_event'])}, "
        f"`is_earthquake` {pct(tr['is_earthquake'])}, `is_work_day` {pct(tr['is_work_day'])}.")
    tre = hol["transferred_rows"]
    if len(tre):
        add("")
        add("Holidays treated as NORMAL days because they were transferred:")
        add("")
        add("| date | locale | place | description |")
        add("|---|---|---|---|")
        for _, r in tre.iterrows():
            add(f"| {r['date'].date()} | {r['locale']} | {r['locale_name']} | {r['description']} |")
    add("")
    add("### Zero-sales series, closed stores and outliers  (decision: KEEP all rows, FLAG them)")
    add("- **Why keep?** (1) lags need a complete daily history per series, (2) the final forecast needs a value for every "
        "store x family, (3) every model owner can filter the same way using the same flags.")
    add(f"- All statistics use only data up to **{config.CLEANING_STATS_CUTOFF}** (same as the train end of Sara's split) so nothing leaks.")
    add(f"- `qc_zero_series`: {int(qc['qc_zero_series'].sum())} of {len(qc)} series never sold anything "
        f"({pct(tr['qc_zero_series'])} of rows). Suggestion: do not train on them, predict 0.")
    add(f"- `qc_before_first_sale`: {pct(tr['qc_before_first_sale'])} of rows are before a series' first sale "
        "(store not opened / product not sold yet). Suggestion: exclude from training.")
    add(f"- `qc_store_closed_day`: {pct(tr['qc_store_closed_day'])} of rows are days where the store sold nothing in any family.")
    add(f"- `qc_sales_outlier`: {pct(tr['qc_sales_outlier'])} of rows are above Q3 + {config.OUTLIER_IQR_MULT:g} x IQR of their own series "
        "(non-zero days). **Not removed or clipped**: they are real sales (promotions, big events).")
    add("- **IMPORTANT: `qc_` columns are built from sales, so they are NOT available for the future. "
        "Use them only to filter/inspect rows. NEVER use them as model features.**")
    add("")
    add("## 3. Things the next person must know")
    add("- **Dec 25 is missing in every year** (stores closed). A row-based `shift(16)` is off by one day around it; "
        "build lags by DATE (merge on `date + 16 days`) or re-index to a full calendar.")
    add("- `transactions.csv` is intentionally not merged: it does not exist for future days (it would be a leak).")
    add("- Fit encoders/scalers on the training period only (Adham), split by date, never shuffle (Sara).")
    add("- Load with: `pd.read_parquet('data/processed/train_clean.parquet')`")
    add("")
    add("## 4. Column dictionary")
    add("")
    add("| column | meaning |")
    add("|---|---|")
    for col in config.TRAIN_CLEAN_COLUMNS:
        add(f"| `{col}` | {config.COLUMN_DOC[col]} |")
    add("")
    config.CLEANING_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.CLEANING_REPORT_PATH.write_text("\n".join(L), encoding="utf-8")


# ----------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    setup_console()
    ap = argparse.ArgumentParser(description="StockWise AI: clean the raw data.")
    ap.add_argument("--csv", action="store_true", help="also save train_clean.csv")
    ap.add_argument("--skip-validation", action="store_true", help="do not run validate_data.py at the end")
    args = ap.parse_args(argv)

    try:
        import pyarrow  # noqa: F401
    except ImportError:
        print("pyarrow is missing. Run:  pip install pyarrow   then run this script again.")
        return 1
    missing = missing_required_files()
    if missing:
        print("Required files are missing from data/raw: " + ", ".join(p.name for p in missing))
        return 1

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading raw files ...")
    train = load_raw("train")
    stores_raw, oil_raw, hol_raw = load_raw("stores"), load_raw("oil"), load_raw("holidays")
    test = load_raw("test") if config.RAW_FILES["test"].exists() else None

    validate_sales_table(train, "train.csv", has_sales=True)
    if test is not None:
        validate_sales_table(test, "test.csv", has_sales=False)
    stores = clean_stores(stores_raw)
    absent = set(train["store_nbr"]) - set(stores["store_nbr"])
    if absent:
        raise ValueError(f"train has stores that are not in stores.csv: {sorted(absent)}")
    train = downcast_sales_table(train, has_sales=True)
    if test is not None:
        test = downcast_sales_table(test, has_sales=False)

    all_dates = train["date"] if test is None else pd.concat([train["date"], test["date"]])
    span_start, span_end = all_dates.min(), all_dates.max()

    print("[2/6] Oil: one price per day, carry last known price forward ...")
    oil_daily, oil_info = build_oil_daily(oil_raw, span_start, span_end)

    print("[3/6] Holidays: separating real holidays / events / transferred ...")
    hol, hol_info = classify_holidays(hol_raw)
    calendar = build_calendar(all_dates, stores, hol)
    unmatched = sorted(
        {f"{r.locale_name} ({r.locale})" for r in hol.itertuples()
         if (r.locale == "Regional" and r.locale_name not in set(stores["state"]))
         or (r.locale == "Local" and r.locale_name not in set(stores["city"]))}
    )

    print("[4/6] Merging stores + oil + holiday flags onto the sales rows ...")
    train_clean = enrich(train, stores, oil_daily, calendar)

    print("[5/6] Flagging zero series, closed stores and outliers ...")
    train_clean, series_summary = add_qc_flags(train_clean, config.CLEANING_STATS_CUTOFF)
    train_clean = train_clean[config.TRAIN_CLEAN_COLUMNS]
    if len(train_clean) != len(train):
        raise RuntimeError("row count changed during cleaning")

    test_clean = None
    if test is not None:
        test_clean = enrich(test, stores, oil_daily, calendar)[config.TEST_CLEAN_COLUMNS]

    print("[6/6] Saving ...")
    train_clean.to_parquet(config.TRAIN_CLEAN_PATH, index=False)
    if test_clean is not None:
        test_clean.to_parquet(config.TEST_CLEAN_PATH, index=False)
    if args.csv:
        train_clean.to_csv(config.PROCESSED_DIR / "train_clean.csv", index=False, date_format="%Y-%m-%d")
    hol.to_csv(config.HOLIDAYS_CLEAN_PATH, index=False, date_format="%Y-%m-%d", encoding="utf-8")
    oil_daily.to_csv(config.OIL_DAILY_PATH, index=False, date_format="%Y-%m-%d")
    series_summary.to_csv(config.SERIES_SUMMARY_PATH, index=False, date_format="%Y-%m-%d")

    write_cleaning_report({
        "train_clean": train_clean, "test_clean": test_clean, "oil_info": oil_info,
        "hol_info": hol_info, "unmatched": unmatched, "qc": series_summary,
    })
    print(f"\nSaved: {config.TRAIN_CLEAN_PATH}  ({len(train_clean):,} rows)")
    if test_clean is not None:
        print(f"Saved: {config.TEST_CLEAN_PATH}  ({len(test_clean):,} rows)")
    print(f"Saved: {config.CLEANING_REPORT_PATH}")

    if args.skip_validation:
        return 0
    from validate_data import run_validation
    return 0 if run_validation() else 1


if __name__ == "__main__":
    raise SystemExit(main())

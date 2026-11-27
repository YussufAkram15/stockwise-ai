"""Level 2: turn the 16-day forecast into reorder advice for every store x family.

Run from the project root (after the forecast exists: python src/forecast_best_model.py):
    python src/inventory.py

Reads  : reports/forecast_16d.parquet     next 16 days: date, store_nbr, family, forecast
         reports/val_predictions.parquet  July 2017: date, store_nbr, family, sales, prediction
Writes : reports/reorder_table.csv        one row per store x family with the advice
         reports/inventory_summary.md     assumptions, counts, most urgent items

The formulas (docs: technical guide, section 11)
------------------------------------------------
    sigma          = standard deviation of the daily forecast error in validation
    safety_stock   = Z * sigma * sqrt(lead_time)          Z = 1.65 -> about 95% service level
    reorder_point  = average daily forecast * lead_time + safety_stock
    order_up_to    = 16-day forecast + safety_stock
    status         = REORDER    if on_hand <= reorder_point   (and the 16-day forecast is above 0)
                     OVERSTOCK  if on_hand >  order_up_to
                     OK         otherwise
                     (a series forecast at 0 by the zero rule in forecast_best_model.py
                      is never REORDER: nothing is expected to sell, so nothing is ordered)
    order_qty      = order_up_to - on_hand   (only when REORDER)

IMPORTANT: the Favorita data has NO stock levels and NO lead times. Both are
SIMULATED below (fixed random seed, so every run gives the same numbers).
The table shows how the method works; it is not real advice for Favorita.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import config
from data_io import setup_console

KEYS = ["store_nbr", "family"]
HORIZON = 16

# Business assumptions (change them here, and only here)
Z = 1.65                       # 95% cycle service level: stock runs short in about 1 order cycle in 20
LEAD_TIME_DAYS = (2, 7)        # simulated delivery time per store, in days (min, max)
COVER_DAYS = (0, 30)           # simulated stock on the shelf, in days of expected sales
SEED = 42                      # fixed random seed -> same simulation every run

FORECAST_PATH = config.REPORTS_DIR / "forecast_16d.parquet"
VAL_PREDICTIONS_PATH = config.REPORTS_DIR / "val_predictions.parquet"
REORDER_TABLE_PATH = config.REPORTS_DIR / "reorder_table.csv"
SUMMARY_PATH = config.REPORTS_DIR / "inventory_summary.md"


# ---------------------------------------------------------------------------
# Step 1: forecast per series
# ---------------------------------------------------------------------------
def load_forecast() -> pd.DataFrame:
    fc = pd.read_parquet(FORECAST_PATH)
    fc["forecast"] = fc["forecast"].fillna(0).clip(lower=0)   # no negative sales
    n_series = fc[KEYS].drop_duplicates().shape[0]
    n_days = fc["date"].nunique()
    if n_days != HORIZON:
        raise ValueError(f"forecast must cover {HORIZON} days, found {n_days}")
    print(f"Forecast: {n_series} series x {n_days} days, "
          f"{fc['date'].min().date()} -> {fc['date'].max().date()}, model = {fc['model'].iloc[0]}")
    return fc


# ---------------------------------------------------------------------------
# Step 2: how wrong is the forecast usually? (sigma, from validation errors)
# ---------------------------------------------------------------------------
def forecast_error_sigma() -> pd.DataFrame:
    val = pd.read_parquet(VAL_PREDICTIONS_PATH)
    val["error"] = val["sales"] - val["prediction"]
    sigma = val.groupby(KEYS)["error"].std().rename("sigma").reset_index()
    # A series without enough errors gets the typical sigma of its family
    sigma["sigma"] = sigma["sigma"].fillna(sigma.groupby("family")["sigma"].transform("median"))
    sigma["sigma"] = sigma["sigma"].fillna(0)
    return sigma


# ---------------------------------------------------------------------------
# Step 3: simulate what the data does not have (lead time, stock on hand)
# ---------------------------------------------------------------------------
def simulate_stock(table: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    stores = np.sort(table["store_nbr"].unique())
    lead = rng.integers(LEAD_TIME_DAYS[0], LEAD_TIME_DAYS[1] + 1, size=len(stores))
    table["lead_time_days"] = table["store_nbr"].map(dict(zip(stores, lead)))

    cover = rng.uniform(COVER_DAYS[0], COVER_DAYS[1], size=len(table))
    table["on_hand"] = (cover * table["avg_daily_forecast"]).round(1)
    return table


# ---------------------------------------------------------------------------
# Step 4 + 5: formulas and flags
# ---------------------------------------------------------------------------
def apply_policy(table: pd.DataFrame) -> pd.DataFrame:
    table["safety_stock"] = Z * table["sigma"] * np.sqrt(table["lead_time_days"])
    table["reorder_point"] = table["avg_daily_forecast"] * table["lead_time_days"] + table["safety_stock"]
    table["order_up_to"] = table["forecast_16d"] + table["safety_stock"]

    # No forecast sales (zero rule) -> no order, even if a safety stock was computed from July errors
    reorder = (table["forecast_16d"] > 0) & (table["on_hand"] <= table["reorder_point"])
    overstock = table["on_hand"] > table["order_up_to"]
    table["status"] = np.select([reorder, overstock], ["REORDER", "OVERSTOCK"], default="OK")
    table["order_qty"] = np.where(reorder, np.ceil(table["order_up_to"] - table["on_hand"]), 0)
    return table


def add_stockout_date(table: pd.DataFrame, fc: pd.DataFrame) -> pd.DataFrame:
    """First forecast day on which the expected sales so far exceed the stock on hand."""
    fc = fc.sort_values(KEYS + ["date"]).merge(table[KEYS + ["on_hand"]], on=KEYS)
    fc["cum_forecast"] = fc.groupby(KEYS)["forecast"].cumsum()
    short = fc[fc["cum_forecast"] > fc["on_hand"]]
    first = short.groupby(KEYS)["date"].min().rename("stockout_date").reset_index()
    return table.merge(first, on=KEYS, how="left")


# ---------------------------------------------------------------------------
# Step 6: save the table and a short summary
# ---------------------------------------------------------------------------
def write_summary(table: pd.DataFrame, fc: pd.DataFrame) -> None:
    counts = table["status"].value_counts().reindex(["REORDER", "OK", "OVERSTOCK"], fill_value=0)
    urgent = table[table["status"] == "REORDER"].sort_values(["stockout_date", "order_qty"],
                                                             ascending=[True, False]).head(10)
    by_family = (table[table["status"] == "REORDER"].groupby("family").size()
                 .sort_values(ascending=False).head(5))

    lines = [
        "# Inventory summary (auto-generated by src/inventory.py)",
        "",
        f"Forecast model: `{fc['model'].iloc[0]}` · horizon {fc['date'].min().date()} → "
        f"{fc['date'].max().date()} ({HORIZON} days) · {len(table):,} store × family series",
        "",
        "## Assumptions (stock and lead times are SIMULATED: the data has none)",
        f"- Service level 95% → Z = {Z}",
        f"- Lead time: random whole number of days per store, {LEAD_TIME_DAYS[0]}–{LEAD_TIME_DAYS[1]}",
        f"- Stock on hand: random {COVER_DAYS[0]}–{COVER_DAYS[1]} days of expected sales per series",
        "- σ (forecast error) = standard deviation of the daily errors in validation (July 2017)",
        f"- Random seed {SEED}: every run gives the same simulation",
        "",
        "## Result",
        "| Status | Series |",
        "|---|---|",
        *[f"| {s} | {n:,} |" for s, n in counts.items()],
        "",
        "## Ten most urgent (earliest expected stockout)",
        "| Store | Family | On hand | Reorder point | Runs out on | Order qty |",
        "|---|---|---|---|---|---|",
        *[f"| {r.store_nbr} | {r.family} | {r.on_hand:,.1f} | {r.reorder_point:,.1f} | "
          f"{r.stockout_date.date() if pd.notna(r.stockout_date) else '-'} | {r.order_qty:,.0f} |"
          for r in urgent.itertuples()],
        "",
        "## Families with the most REORDER flags",
        *[f"- {fam}: {n} stores" for fam, n in by_family.items()],
        "",
        "Slow sellers (a few units a day, sold irregularly) get REORDER often: their safety",
        "stock is large compared with their daily sales, because their forecast error is large",
        "compared with their sales. A series forecast at 0 (no sales in the last 28 days, the",
        "zero rule in forecast_best_model.py) is never REORDER: nothing is expected to sell.",
        "",
    ]
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    setup_console()
    for path in (FORECAST_PATH, VAL_PREDICTIONS_PATH):
        if not path.exists():
            print(f"{path.name} not found. Run first:  python src/forecast_best_model.py")
            return 1

    fc = load_forecast()
    table = (fc.groupby(KEYS)["forecast"]
             .agg(forecast_16d="sum", avg_daily_forecast="mean").reset_index())
    table = table.merge(forecast_error_sigma(), on=KEYS, how="left")
    table["sigma"] = table["sigma"].fillna(0)

    table = simulate_stock(table)
    table = apply_policy(table)
    table = add_stockout_date(table, fc)

    columns = ["store_nbr", "family", "status", "order_qty", "on_hand", "reorder_point",
               "safety_stock", "lead_time_days", "avg_daily_forecast", "forecast_16d",
               "sigma", "order_up_to", "stockout_date"]
    table["status"] = pd.Categorical(table["status"], ["REORDER", "OK", "OVERSTOCK"], ordered=True)
    table = table[columns].sort_values(["status", "stockout_date", "store_nbr", "family"])
    numbers = table.select_dtypes("number").columns
    table[numbers] = table[numbers].round(1)
    table.to_csv(REORDER_TABLE_PATH, index=False)
    write_summary(table, fc)

    print(table["status"].value_counts().to_string())
    print(f"Saved: {REORDER_TABLE_PATH.relative_to(config.PROJECT_ROOT)}")
    print(f"Saved: {SUMMARY_PATH.relative_to(config.PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

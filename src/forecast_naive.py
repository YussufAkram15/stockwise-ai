"""Level 2 input: write the 16-day forecast with the naive lag-21 baseline.

Run from the project root:
    python src/forecast_naive.py

Why this script exists
----------------------
The inventory layer (src/inventory.py) needs two files. Any model can write them;
this script writes them with the naive baseline so Level 2 can be built now.
When the team picks the best model, that model writes the SAME two files
(same names, same columns) and nothing else has to change.

    reports/forecast_16d.parquet     the next 16 days, one row per store x family x day
        columns: date, store_nbr, family, forecast, model
    reports/val_predictions.parquet  the same model on the validation month (July 2017)
        columns: date, store_nbr, family, sales, prediction
        -> used to measure how wrong the forecast usually is (sigma)

It also writes reports/recent_sales.parquet (date, store_nbr, family, sales):
the last 60 real days before the forecast, so the dashboard can draw history
next to the forecast without loading the big raw file.

"Today" (day 0) is the last day of train.csv, 2017-08-15. The forecast covers
2017-08-16 .. 2017-08-31: the same 16 days as Kaggle's test.csv.
The forecast for day d is the sales on day d - 21. The oldest forecast day
(31 Aug) uses 10 Aug, which is before day 0, so nothing from the future is used.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import config
from data_io import setup_console

LAG = 21
HORIZON = 16
KEYS = ["store_nbr", "family"]

FORECAST_PATH = config.REPORTS_DIR / "forecast_16d.parquet"
VAL_PREDICTIONS_PATH = config.REPORTS_DIR / "val_predictions.parquet"
RECENT_SALES_PATH = config.REPORTS_DIR / "recent_sales.parquet"
RECENT_DAYS = 60


def naive_prediction(sales: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """For every series and every date in `dates`, predict the sales of 21 days earlier."""
    series = sales[KEYS].drop_duplicates()
    grid = series.merge(pd.DataFrame({"date": dates}), how="cross")
    grid["source_date"] = grid["date"] - pd.Timedelta(days=LAG)
    past = sales.rename(columns={"date": "source_date", "sales": "prediction"})
    out = grid.merge(past, on=KEYS + ["source_date"], how="left")
    return out.drop(columns="source_date")


def main() -> int:
    setup_console()
    sales = pd.read_csv(
        config.RAW_FILES["train"],
        usecols=["date", "store_nbr", "family", "sales"],
        parse_dates=["date"],
    )

    day0 = pd.Timestamp(config.TEST_END)
    future_dates = pd.date_range(day0 + pd.Timedelta(days=1), periods=HORIZON, freq="D")
    val_dates = pd.date_range(config.VAL_START, config.VAL_END, freq="D")

    # 1. The next 16 days (no real sales exist for them)
    forecast = naive_prediction(sales, future_dates).rename(columns={"prediction": "forecast"})
    forecast["model"] = f"naive_lag{LAG}"

    # 2. The validation month, with the real sales next to the prediction
    val = naive_prediction(sales, val_dates).merge(sales, on=KEYS + ["date"], how="left")
    val = val[["date", "store_nbr", "family", "sales", "prediction"]]

    for name, table in [("forecast", forecast["forecast"]), ("validation", val["prediction"])]:
        missing = int(table.isna().sum())
        if missing:
            print(f"ERROR: {missing} {name} rows have no value 21 days earlier.")
            return 1

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    forecast.to_parquet(FORECAST_PATH, index=False)
    val.to_parquet(VAL_PREDICTIONS_PATH, index=False)
    recent = sales[sales["date"] > day0 - pd.Timedelta(days=RECENT_DAYS)]
    recent.to_parquet(RECENT_SALES_PATH, index=False)

    print(f"Day 0 (today): {day0.date()}")
    print(f"Forecast: {future_dates[0].date()} -> {future_dates[-1].date()}, "
          f"{len(forecast):,} rows  -> {FORECAST_PATH.relative_to(config.PROJECT_ROOT)}")
    print(f"Validation predictions: {len(val):,} rows  -> "
          f"{VAL_PREDICTIONS_PATH.relative_to(config.PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

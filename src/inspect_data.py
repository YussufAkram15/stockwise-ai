"""STEP 1 (Zenab): look at the raw files BEFORE cleaning anything.

Run from the project root:   python src/inspect_data.py
Nothing is modified. The result is printed and saved to reports/inspection_report.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import config
from data_io import Report, load_raw, missing_required_files, setup_console


def describe_table(rep: Report, name: str, df: pd.DataFrame) -> None:
    rep.section(f"{name}  ({config.RAW_FILES[name].name})")
    rep.add(f"shape        : {df.shape[0]:,} rows x {df.shape[1]} columns")
    rep.add("columns      : " + ", ".join(f"{c} [{t}]" for c, t in df.dtypes.astype(str).items()))
    nulls = df.isna().sum()
    nulls = nulls[nulls > 0]
    rep.add("missing      : " + ("none" if nulls.empty else ", ".join(f"{c}={n:,}" for c, n in nulls.items())))
    rep.add(f"duplicates   : {int(df.duplicated().sum()):,} fully duplicated rows")
    if "date" in df.columns:
        rep.add(f"date range   : {df['date'].min().date()} -> {df['date'].max().date()} "
                f"({df['date'].nunique():,} unique dates)")


def inspect_train(rep: Report, train: pd.DataFrame, issues: list[str]) -> None:
    n_dates = train["date"].nunique()
    n_stores = train["store_nbr"].nunique()
    n_fam = train["family"].nunique()
    expected_rows = n_dates * n_stores * n_fam

    rep.add("")
    rep.add("-- train details --")
    rep.add(f"stores / families / dates : {n_stores} / {n_fam} / {n_dates}")
    rep.add(f"full grid would be        : {expected_rows:,} rows;  actual: {len(train):,} rows")
    if expected_rows != len(train):
        issues.append("train is not a complete date x store x family grid")

    all_days = pd.date_range(train["date"].min(), train["date"].max(), freq="D")
    missing_days = all_days.difference(pd.DatetimeIndex(train["date"].unique()))
    only_dec25 = all(d.month == 12 and d.day == 25 for d in missing_days)
    shown = ", ".join(str(d.date()) for d in missing_days[:12])
    rep.add(f"calendar days with NO rows : {len(missing_days)}  ({shown}{' ...' if len(missing_days) > 12 else ''})")
    rep.add("   -> " + ("all are Dec 25 (stores closed) - expected, not an error" if only_dec25
                         else "NOT only Dec 25 - please look at this"))

    dup_key = int(train.duplicated(["date", "store_nbr", "family"]).sum())
    rep.add(f"duplicate (date, store, family) keys : {dup_key:,}")
    if dup_key:
        issues.append("train has duplicate (date, store_nbr, family) rows")

    neg = int((train["sales"] < 0).sum())
    rep.add(f"sales  : min={train['sales'].min():,.2f}  max={train['sales'].max():,.2f}  "
            f"mean={train['sales'].mean():,.2f}  zero share={(train['sales'] == 0).mean():.1%}  negative rows={neg:,}")
    if neg:
        issues.append("train has negative sales")
    rep.add(f"onpromotion : min={train['onpromotion'].min()}  max={train['onpromotion'].max()}  "
            f"share of rows with promo={(train['onpromotion'] > 0).mean():.1%}")
    if train.isna().any().any():
        issues.append("train has missing values")


def inspect_test(rep: Report, test: pd.DataFrame, train: pd.DataFrame, issues: list[str]) -> None:
    rep.add("")
    rep.add("-- test details --")
    days = test["date"].nunique()
    rep.add(f"forecast horizon : {days} days  (starts {test['date'].min().date()}, "
            f"train ends {train['date'].max().date()})")
    if test["date"].min() <= train["date"].max():
        issues.append("test dates overlap train dates")
    if set(test["id"]) & set(train["id"]):
        issues.append("test ids overlap train ids")
    s_train = set(zip(train["store_nbr"], train["family"]))
    s_test = set(zip(test["store_nbr"], test["family"]))
    rep.add(f"series in test but not in train : {len(s_test - s_train)}")
    if s_test - s_train:
        issues.append("test has store x family series that never appear in train")


def inspect_oil(rep: Report, oil: pd.DataFrame, train: pd.DataFrame, issues: list[str]) -> None:
    rep.add("")
    rep.add("-- oil details --")
    rep.add(f"duplicate dates          : {int(oil['date'].duplicated().sum())}")
    if oil["date"].duplicated().any():
        issues.append("oil.csv has duplicate dates")
    span = pd.date_range(train["date"].min(), train["date"].max(), freq="D")
    have = oil.dropna(subset=["dcoilwtico"])["date"]
    rep.add(f"train days without a real oil price : {len(span.difference(pd.DatetimeIndex(have))):,} "
            f"of {len(span):,}  (weekends/holidays/blanks - will be carried forward)")
    first = oil["date"].min()
    first_nan = bool(oil.loc[oil["date"] == first, "dcoilwtico"].isna().iloc[0])
    rep.add(f"first oil row ({first.date()}) has no price : {first_nan}  (will be back-filled from the first real price)")
    rep.add(f"price range              : {oil['dcoilwtico'].min():.2f} -> {oil['dcoilwtico'].max():.2f}")
    if (oil["dcoilwtico"].dropna() <= 0).any():
        issues.append("oil has zero/negative prices")


def inspect_holidays(rep: Report, hol: pd.DataFrame, stores: pd.DataFrame) -> None:
    rep.add("")
    rep.add("-- holidays details --")
    rep.add("rows by type   : " + ", ".join(f"{k}={v}" for k, v in hol["type"].value_counts().items()))
    rep.add("rows by locale : " + ", ".join(f"{k}={v}" for k, v in hol["locale"].value_counts().items()))
    rep.add(f"transferred=True rows : {int(hol['transferred'].sum())}  (the holiday was moved - NOT a day off on that date)")
    unknown_types = sorted(set(hol["type"]) - set(config.KNOWN_HOLIDAY_TYPES))
    rep.add(f"unknown types : {unknown_types if unknown_types else 'none'}")
    reg = set(hol.loc[hol["locale"] == "Regional", "locale_name"]) - set(stores["state"])
    loc = set(hol.loc[hol["locale"] == "Local", "locale_name"]) - set(stores["city"])
    rep.add(f"regional holidays in states with no store : {sorted(reg) if reg else 'none'}")
    rep.add(f"local holidays in cities with no store     : {sorted(loc) if loc else 'none'}")


def inspect_stores(rep: Report, stores: pd.DataFrame, train: pd.DataFrame, issues: list[str]) -> None:
    rep.add("")
    rep.add("-- stores details --")
    rep.add(f"unique store_nbr : {stores['store_nbr'].nunique()} (rows: {len(stores)})")
    rep.add(f"cities / states / types / clusters : {stores['city'].nunique()} / {stores['state'].nunique()} / "
            f"{stores['type'].nunique()} / {stores['cluster'].nunique()}")
    if stores["store_nbr"].duplicated().any():
        issues.append("stores.csv has duplicate store_nbr")
    absent = set(train["store_nbr"]) - set(stores["store_nbr"])
    rep.add(f"train stores missing from stores.csv : {sorted(absent) if absent else 'none'}")
    if absent:
        issues.append("some train stores are missing from stores.csv")


def main() -> int:
    setup_console()
    rep = Report()
    issues: list[str] = []

    rep.section("STOCKWISE AI - RAW DATA INSPECTION")
    rep.add(f"project root : {config.PROJECT_ROOT}")
    for name, path in config.RAW_FILES.items():
        size = f"{path.stat().st_size / 1e6:,.1f} MB" if path.exists() else "MISSING"
        rep.add(f"  {path.name:<24} {size}")

    missing = missing_required_files()
    if missing:
        rep.add("")
        rep.add("STOP: required files are missing from data/raw: " + ", ".join(p.name for p in missing))
        return 1

    tables = {n: load_raw(n) for n in config.RAW_FILES if config.RAW_FILES[n].exists()}
    for name in ("train", "test", "stores", "oil", "holidays", "transactions", "sample_submission"):
        if name not in tables:
            continue
        describe_table(rep, name, tables[name])
        if name == "train":
            inspect_train(rep, tables["train"], issues)
        elif name == "test":
            inspect_test(rep, tables["test"], tables["train"], issues)
        elif name == "oil":
            inspect_oil(rep, tables["oil"], tables["train"], issues)
        elif name == "holidays":
            inspect_holidays(rep, tables["holidays"], tables["stores"])
        elif name == "stores":
            inspect_stores(rep, tables["stores"], tables["train"], issues)
        elif name == "transactions":
            rep.add("")
            rep.add("-- note: transactions.csv is NOT used in the clean file: it does not exist for the "
                    "future (test) days, so using it would be a leak.")
        elif name == "sample_submission" and "test" in tables:
            same = set(tables["sample_submission"]["id"]) == set(tables["test"]["id"])
            rep.add("")
            rep.add(f"-- sample_submission ids match test ids : {same}")
            if not same:
                issues.append("sample_submission ids do not match test ids")

    rep.section("SUMMARY")
    if issues:
        rep.add("Problems that need attention BEFORE cleaning:")
        for i in issues:
            rep.add(f"  - {i}")
    else:
        rep.add("No blocking problems found in the raw files.")
    rep.add("Known quirks that prepare_data.py handles: oil blanks, transferred holidays, Dec 25 gap.")

    rep.save(config.INSPECTION_REPORT_PATH)
    print(f"\nSaved: {config.INSPECTION_REPORT_PATH}")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())

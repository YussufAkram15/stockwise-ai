"""PROOF that the cleaned data is correct.

Run from the project root:   python src/validate_data.py
(prepare_data.py also runs it automatically at the end.)

It re-reads the RAW files and the CLEANED files and checks, one by one, that nothing
was lost, changed or leaked. Every line is PASS / FAIL / WARN:
  FAIL = a real problem (exit code 1)      WARN = unusual, please read it
The result is printed and saved to reports/validation_report.txt
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import config
from data_io import Report, load_raw, setup_console


class Checker:
    def __init__(self, rep: Report) -> None:
        self.rep = rep
        self.n = {"PASS": 0, "FAIL": 0, "WARN": 0}

    def check(self, name: str, condition, detail: str = "", level: str = "FAIL") -> bool:
        ok = bool(condition)
        status = "PASS" if ok else level
        self.n[status] += 1
        self.rep.add(f"[{status}] {name}" + (f"   ({detail})" if detail else ""))
        return ok

    @property
    def ok(self) -> bool:
        return self.n["FAIL"] == 0


def _expected_flags(sd: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    """Independent re-computation of 'does a holiday row apply to this store on this date'
    (plain Python sets, different code path from prepare_data.py)."""
    nat = set(rows.loc[rows["locale"] == "National", "date"])
    reg = set(zip(rows.loc[rows["locale"] == "Regional", "date"], rows.loc[rows["locale"] == "Regional", "locale_name"]))
    loc = set(zip(rows.loc[rows["locale"] == "Local", "date"], rows.loc[rows["locale"] == "Local", "locale_name"]))
    return np.array(
        [(d in nat) or ((d, s) in reg) or ((d, c) in loc)
         for d, s, c in zip(sd["date"], sd["state"], sd["city"])],
        dtype=bool,
    )


def _expected_scope(sd: pd.DataFrame, rows: pd.DataFrame, scope: str) -> np.ndarray:
    r = rows[rows["locale"] == scope]
    if scope == "National":
        dates = set(r["date"])
        return np.array([d in dates for d in sd["date"]], dtype=bool)
    col = "state" if scope == "Regional" else "city"
    pairs = set(zip(r["date"], r["locale_name"]))
    return np.array([(d, p) in pairs for d, p in zip(sd["date"], sd[col])], dtype=bool)


def validate_train(c: Checker, rep: Report) -> pd.DataFrame | None:
    rep.section("A. Cleaned train file: structure")
    if not c.check("train_clean.parquet exists", config.TRAIN_CLEAN_PATH.exists(), str(config.TRAIN_CLEAN_PATH.name)):
        return None
    clean = pd.read_parquet(config.TRAIN_CLEAN_PATH)
    raw = load_raw("train")
    stores = load_raw("stores")
    oil = load_raw("oil")
    hol = load_raw("holidays")
    key = ["store_nbr", "family"]

    c.check("columns are exactly the agreed schema (names and order)",
            list(clean.columns) == config.TRAIN_CLEAN_COLUMNS,
            f"{clean.shape[1]} columns")
    c.check("date column is a real datetime", pd.api.types.is_datetime64_any_dtype(clean["date"]))
    c.check("sales is numeric", pd.api.types.is_numeric_dtype(clean["sales"]))

    rep.section("B. Nothing lost, added or changed compared with the raw train.csv")
    c.check("same number of rows as train.csv", len(clean) == len(raw), f"raw={len(raw):,} clean={len(clean):,}")
    c.check("id is unique", clean["id"].is_unique)
    c.check("(date, store_nbr, family) is unique", not clean.duplicated(["date", "store_nbr", "family"]).any())
    r = raw.sort_values("id").reset_index(drop=True)
    k = clean.sort_values("id").reset_index(drop=True)
    same_ids = np.array_equal(r["id"].to_numpy(), k["id"].to_numpy())
    c.check("same set of ids as train.csv", same_ids)
    if same_ids:
        for col in ["date", "store_nbr", "family", "sales", "onpromotion"]:
            c.check(f"column '{col}' identical to train.csv row by row",
                    np.array_equal(r[col].to_numpy(), k[col].to_numpy()))
    del r, k

    rep.section("C. Missing / invalid values")
    nulls = clean.isna().sum()
    c.check("no missing values in any column", not nulls.any(),
            "none" if not nulls.any() else str(nulls[nulls > 0].to_dict()))
    c.check("sales >= 0 and finite", bool((clean["sales"] >= 0).all() and np.isfinite(clean["sales"]).all()))
    c.check("onpromotion >= 0", bool((clean["onpromotion"] >= 0).all()))

    rep.section("D. Complete calendar grid (needed for lags / rolling averages)")
    n_series = clean[key].drop_duplicates().shape[0]
    n_dates = clean["date"].nunique()
    per_date = clean.groupby("date").size()
    per_series = clean.groupby(key).size()
    c.check("every date has one row for EVERY store x family", per_date.nunique() == 1 and per_date.iloc[0] == n_series,
            f"{n_series} series, rows per date min={per_date.min()} max={per_date.max()}")
    c.check("every series has the same number of days", per_series.nunique() == 1 and per_series.iloc[0] == n_dates,
            f"{n_dates} days per series")
    all_days = pd.date_range(clean["date"].min(), clean["date"].max(), freq="D")
    gaps = all_days.difference(pd.DatetimeIndex(clean["date"].unique()))
    c.check("the only missing calendar days are Dec 25", all(d.month == 12 and d.day == 25 for d in gaps),
            f"{len(gaps)} missing days", level="WARN")
    step = clean.groupby(key)["date"].diff().dropna()
    c.check("inside each series dates are strictly increasing (sorted)", bool((step > pd.Timedelta(0)).all()))

    rep.section("E. Stores merge")
    attrs = clean[["store_nbr", "city", "state", "store_type", "cluster"]].drop_duplicates()
    c.check("each store has exactly ONE city/state/type/cluster", len(attrs) == clean["store_nbr"].nunique())
    ref = stores.rename(columns={"type": "store_type"})[["store_nbr", "city", "state", "store_type", "cluster"]]
    cmp = attrs.merge(ref, on="store_nbr", how="left", suffixes=("", "_ref"))
    same = all((cmp[col].astype(str) == cmp[col + "_ref"].astype(str)).all() for col in ["city", "state", "store_type", "cluster"])
    c.check("store attributes equal stores.csv", same)

    rep.section("F. Oil (independent re-computation from oil.csv)")
    c.check("oil_price has no missing values and is > 0", bool(clean["oil_price"].notna().all() and (clean["oil_price"] > 0).all()),
            f"min={clean['oil_price'].min():.2f} max={clean['oil_price'].max():.2f}")
    daily = clean.drop_duplicates("date").set_index("date").sort_index()
    c.check("one oil price per date (same for all stores)",
            clean.groupby("date")["oil_price"].nunique().max() == 1)
    s = oil.drop_duplicates("date").set_index("date")["dcoilwtico"].sort_index()
    idx = pd.date_range(min(s.index.min(), daily.index.min()), max(s.index.max(), daily.index.max()), freq="D")
    expected = s.reindex(idx).ffill().bfill().reindex(daily.index)
    c.check("oil_price equals 'carry last known price forward' recomputed from oil.csv",
            np.allclose(expected.to_numpy(), daily["oil_price"].to_numpy()))
    real_dates = set(s.dropna().index)
    exp_filled = np.array([d not in real_dates for d in daily.index])
    c.check("oil_is_filled = 1 exactly on days without a real price",
            np.array_equal(exp_filled, daily["oil_is_filled"].to_numpy().astype(bool)),
            f"{int(exp_filled.sum())} filled days of {len(daily)}")
    on_real = daily.index.isin(list(real_dates))
    c.check("on days with a real price, the value is untouched",
            np.allclose(s.reindex(daily.index[on_real]).to_numpy(), daily.loc[on_real, "oil_price"].to_numpy()))

    rep.section("G. Holiday flags (independent re-computation from holidays_events.csv)")
    flags = config.CALENDAR_FLAGS
    c.check("all holiday flags contain only 0 and 1", all(set(clean[f].unique()) <= {0, 1} for f in flags))
    c.check("flags are the same for all families of a store on a day",
            len(clean[["date", "store_nbr"] + flags].drop_duplicates()) == clean[["date", "store_nbr"]].drop_duplicates().shape[0])
    sd = clean.drop_duplicates(["date", "store_nbr"])[["date", "store_nbr", "city", "state"] + flags].reset_index(drop=True)
    real = hol[hol["type"].isin(config.REAL_HOLIDAY_TYPES) & (~hol["transferred"])]
    for scope, col in [("National", "is_holiday_national"), ("Regional", "is_holiday_regional"), ("Local", "is_holiday_local")]:
        exp = _expected_scope(sd, real, scope)
        c.check(f"{col} matches the real holidays of scope {scope}",
                np.array_equal(exp, sd[col].to_numpy().astype(bool)), f"{int(exp.sum()):,} store-days")
    c.check("is_holiday = national OR regional OR local",
            np.array_equal(sd[["is_holiday_national", "is_holiday_regional", "is_holiday_local"]].max(axis=1).to_numpy(),
                           sd["is_holiday"].to_numpy()))
    events = hol[(hol["type"] == config.EVENT_TYPE) & (~hol["transferred"])]
    c.check("is_event matches the Event rows", np.array_equal(_expected_flags(sd, events), sd["is_event"].to_numpy().astype(bool)))
    quake = events[events["description"].str.contains("terremoto", case=False, na=False)]
    c.check("is_earthquake matches the earthquake rows", np.array_equal(_expected_flags(sd, quake), sd["is_earthquake"].to_numpy().astype(bool)),
            f"{int(sd['is_earthquake'].sum()):,} store-days")
    wd = hol[(hol["type"] == config.WORK_DAY_TYPE) & (~hol["transferred"])]
    c.check("is_work_day matches the Work Day rows", np.array_equal(_expected_flags(sd, wd), sd["is_work_day"].to_numpy().astype(bool)))

    nat_real_dates = set(real.loc[real["locale"] == "National", "date"])
    moved = hol[hol["transferred"] & (hol["locale"] == "National")]
    only_moved = set(moved["date"]) - nat_real_dates
    sub = sd[sd["date"].isin(only_moved)]
    c.check("TRANSFERRED national holidays are treated as normal days",
            bool((sub["is_holiday_national"] == 0).all()), f"{len(only_moved)} transferred dates checked, {sub['date'].nunique()} inside data range")
    trans = set(hol.loc[(hol["type"] == "Transfer") & (~hol["transferred"]) & (hol["locale"] == "National"), "date"])
    sub2 = sd[sd["date"].isin(trans)]
    c.check("the day a national holiday was moved TO is a real holiday for all stores",
            bool((sub2["is_holiday_national"] == 1).all()), f"{sub2['date'].nunique()} dates checked")
    jan1 = sd[(sd["date"].dt.month == 1) & (sd["date"].dt.day == 1)]
    c.check("Jan 1 is a national holiday for all stores (known fact)", bool((jan1["is_holiday_national"] == 1).all()) if len(jan1) else True,
            level="WARN")
    q = sd[sd["date"] == pd.Timestamp("2016-04-16")]
    c.check("2016-04-16 (earthquake day) is flagged (known fact)", bool((q["is_earthquake"] == 1).all()) if len(q) else True, level="WARN")

    rep.section("H. Quality-control flags (zero series, first sale, closed store, outliers)")
    qc = config.QC_FLAGS
    c.check("qc flags contain only 0 and 1", all(set(clean[f].unique()) <= {0, 1} for f in qc))
    cutoff = pd.Timestamp(config.CLEANING_STATS_CUTOFF)
    win = clean[clean["date"] <= cutoff]
    pos_series = set(map(tuple, win.loc[win["sales"] > 0, key].drop_duplicates().to_numpy()))
    all_series = set(map(tuple, clean[key].drop_duplicates().to_numpy()))
    flagged_series = set(map(tuple, clean.loc[clean["qc_zero_series"] == 1, key].drop_duplicates().to_numpy()))
    c.check("qc_zero_series = series with no sale up to the cutoff date", flagged_series == (all_series - pos_series),
            f"{len(flagged_series)} of {len(all_series)} series")
    c.check("qc_zero_series is constant inside each series", clean.groupby(key)["qc_zero_series"].nunique().max() == 1)
    c.check("zero series really have zero sales up to the cutoff", bool((win.loc[win["qc_zero_series"] == 1, "sales"] == 0).all()))
    first = win.loc[win["sales"] > 0].groupby(key)["date"].min().rename("first_pos").reset_index()
    tmp = clean[key + ["date", "qc_before_first_sale"]].merge(first, on=key, how="left")
    exp_before = ((tmp["date"] <= cutoff) & (tmp["first_pos"].isna() | (tmp["date"] < tmp["first_pos"]))).to_numpy()
    c.check("qc_before_first_sale = days before the first sale (up to cutoff)",
            np.array_equal(exp_before, tmp["qc_before_first_sale"].to_numpy().astype(bool)),
            f"{int(exp_before.sum()):,} rows")
    c.check("no sale ever happens on a 'before first sale' row", bool((clean.loc[clean["qc_before_first_sale"] == 1, "sales"] == 0).all()))
    c.check("no 'before first sale' flag after the cutoff", bool((clean.loc[clean["date"] > cutoff, "qc_before_first_sale"] == 0).all()))
    store_day = clean.groupby(["store_nbr", "date"])["sales"].transform("sum")
    c.check("qc_store_closed_day = store sold nothing in any family",
            np.array_equal((store_day == 0).to_numpy(), clean["qc_store_closed_day"].to_numpy().astype(bool)),
            f"{int(clean['qc_store_closed_day'].sum()):,} rows")
    c.check("outliers are real sales (> 0)", bool((clean.loc[clean["qc_sales_outlier"] == 1, "sales"] > 0).all()))
    share = clean["qc_sales_outlier"].mean()
    c.check("outlier share is small (< 5%)", share < 0.05, f"{share:.2%}", level="WARN")

    rep.section("I. Fingerprint of the official Kaggle files (WARN only)")
    e = config.EXPECTED
    c.check("rows = 3,000,888", len(clean) == e["train_rows"], f"{len(clean):,}", level="WARN")
    c.check("54 stores", clean["store_nbr"].nunique() == e["n_stores"], str(clean["store_nbr"].nunique()), level="WARN")
    c.check("33 families", clean["family"].nunique() == e["n_families"], str(clean["family"].nunique()), level="WARN")
    c.check("1684 distinct dates", n_dates == e["n_train_dates"], str(n_dates), level="WARN")
    c.check("date range 2013-01-01 -> 2017-08-15",
            (clean["date"].min() == pd.Timestamp(e["train_start"])) and (clean["date"].max() == pd.Timestamp(e["train_end"])),
            f"{clean['date'].min().date()} -> {clean['date'].max().date()}", level="WARN")
    return clean


def validate_test(c: Checker, rep: Report, train_clean: pd.DataFrame) -> None:
    rep.section("J. Cleaned test file (Kaggle test period)")
    if not config.RAW_FILES["test"].exists():
        c.check("test.csv not in data/raw -> test checks skipped", True)
        return
    if not c.check("test_clean.parquet exists", config.TEST_CLEAN_PATH.exists()):
        return
    raw = load_raw("test")
    clean = pd.read_parquet(config.TEST_CLEAN_PATH)
    c.check("columns are exactly the agreed schema", list(clean.columns) == config.TEST_CLEAN_COLUMNS)
    c.check("same rows, same ids, SAME ORDER as test.csv (needed for the submission file)",
            len(raw) == len(clean) and np.array_equal(raw["id"].to_numpy(), clean["id"].to_numpy()), f"{len(clean):,} rows")
    c.check("date / store / family / onpromotion identical to test.csv",
            all(np.array_equal(raw[col].to_numpy(), clean[col].to_numpy()) for col in ["date", "store_nbr", "family", "onpromotion"]))
    nulls = clean.isna().sum()
    c.check("no missing values", not nulls.any(), "none" if not nulls.any() else str(nulls[nulls > 0].to_dict()))
    c.check("oil_price > 0 on every test day", bool((clean["oil_price"] > 0).all()))
    c.check("all flags contain only 0 and 1", all(set(clean[f].unique()) <= {0, 1} for f in config.CALENDAR_FLAGS))
    s_tr = set(zip(train_clean["store_nbr"], train_clean["family"]))
    s_te = set(zip(clean["store_nbr"], clean["family"]))
    c.check("every test series also exists in train", s_te <= s_tr, f"{len(s_te)} series")
    c.check("test starts after train ends", clean["date"].min() > train_clean["date"].max(),
            f"{clean['date'].min().date()} -> {clean['date'].max().date()}")
    e = config.EXPECTED
    c.check("test range 2017-08-16 -> 2017-08-31 and 28,512 rows",
            clean["date"].min() == pd.Timestamp(e["test_start"]) and clean["date"].max() == pd.Timestamp(e["test_end"]) and len(clean) == e["test_rows"],
            level="WARN")


def run_validation(save_report: bool = True) -> bool:
    setup_console()
    rep = Report()
    c = Checker(rep)
    rep.section("STOCKWISE AI - VALIDATION OF THE CLEANED DATA")
    try:
        train_clean = validate_train(c, rep)
        if train_clean is not None:
            validate_test(c, rep, train_clean)
    except Exception:  # report crashes as a failure instead of a confusing traceback only
        c.n["FAIL"] += 1
        rep.add("[FAIL] validation crashed:")
        rep.add(traceback.format_exc())

    rep.section("FINAL RESULT")
    rep.add(f"PASS = {c.n['PASS']}   WARN = {c.n['WARN']}   FAIL = {c.n['FAIL']}")
    if c.ok:
        rep.add("DATA IS CLEAN: all required checks passed." + ("  (read the WARN lines above)" if c.n["WARN"] else ""))
    else:
        rep.add("PROBLEMS FOUND: fix the FAIL lines above before handing the data to the team.")
    if save_report:
        rep.save(config.VALIDATION_REPORT_PATH)
        print(f"\nSaved: {config.VALIDATION_REPORT_PATH}")
    return c.ok


if __name__ == "__main__":
    raise SystemExit(0 if run_validation() else 1)

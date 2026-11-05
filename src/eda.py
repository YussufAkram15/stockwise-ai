"""EDA (Hazem's part): look at the data first, draw the main charts, write short notes.

Run from the project root (after prepare_data.py):   python src/eda.py

It reads data/processed/train_clean.parquet. `sales` in that file is identical to
train.csv (the validation proves it), the file just has extra columns (store type,
holiday flags) that make the charts easier.

Output (data/reports/eda/):
    01_...png to 09_...png   the charts
    eda_notes.md             plain sentences with REAL numbers from the data
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")  # draw to files only (no window needed, safe on any computer)
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("matplotlib is missing. Run:  pip install matplotlib   then run this script again.")

import config
from data_io import setup_console

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
COLUMNS = ["date", "store_nbr", "family", "sales", "onpromotion", "store_type",
           "is_holiday_national", "is_event", "is_earthquake", "is_work_day", "qc_zero_series"]
QUAKE_DAY = pd.Timestamp("2016-04-16")


def save(fig, name: str) -> None:
    config.EDA_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.EDA_DIR / name, dpi=110)
    plt.close(fig)
    print(f"  saved {name}")


def pct(x: float) -> str:
    return f"{100 * x:+.0f}%"


def main() -> int:
    setup_console()
    if not config.TRAIN_CLEAN_PATH.exists():
        print("train_clean.parquet not found. Run first:  python src/prepare_data.py")
        return 1

    df = pd.read_parquet(config.TRAIN_CLEAN_PATH, columns=COLUMNS)
    notes: list[str] = []
    print(f"Loaded {len(df):,} rows. Drawing charts ...")

    # ---------------- date-level table (all stores and families added up) ----------------
    day = df.groupby("date").agg(
        total=("sales", "sum"),
        nat=("is_holiday_national", "max"),
        ev=("is_event", "max"),
        quake=("is_earthquake", "max"),
        wd=("is_work_day", "max"),
    )
    total = day["total"]
    trend = total.rolling(29, center=True, min_periods=15).mean()
    rel = total / trend            # 1.0 = a normal day of that month (growth is removed)

    # ---------------- 1. total sales over time ----------------
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(total.index, total.values, lw=0.5, color="#9bb7d4", label="daily total")
    ax.plot(trend.index, trend.values, lw=1.8, color="#c0392b", label="29-day average")
    ax.set(title="Total sales per day (all stores, all families)", ylabel="units")
    ax.legend()
    save(fig, "01_total_sales_over_time.png")

    years = total.groupby(total.index.year).agg(["mean", "size"])
    full = years[years["size"] >= 360]["mean"]
    if len(full) >= 2:
        change = full.iloc[-1] / full.iloc[0] - 1
        tail = ("a clear trend, so old and new years should not be treated as equal." if abs(change) >= 0.10
                else "roughly flat over these years.")
        notes.append("Average daily sales (all stores added up) per full year: "
                     + ", ".join(f"{y}: {v:,.0f}" for y, v in full.items())
                     + f"  ({pct(change)} from {full.index[0]} to {full.index[-1]}): " + tail)

    # ---------------- 2. weekday ----------------
    by_wd = total.groupby(total.index.dayofweek).mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(WEEKDAYS, by_wd.values, color=["#5b8db8"] * 5 + ["#e08e45"] * 2)
    ax.set(title="Average daily total sales by weekday", ylabel="units")
    save(fig, "02_sales_by_weekday.png")
    weekend = by_wd.loc[[5, 6]].mean()
    weekdays = by_wd.loc[[0, 1, 2, 3, 4]].mean()
    notes.append(f"Weekends sell {pct(weekend / weekdays - 1)} compared with Monday-Friday; the best day is "
                 f"{WEEKDAYS[int(by_wd.idxmax())]} and the weakest is {WEEKDAYS[int(by_wd.idxmin())]}.")

    # ---------------- 3. day of month (payday) ----------------
    by_dom = rel.groupby(rel.index.day).mean()
    payday = (rel.index.day == 15) | rel.index.is_month_end
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(by_dom.index, by_dom.values, color=["#e08e45" if d in (15, 30, 31) else "#5b8db8" for d in by_dom.index])
    ax.axhline(1, color="grey", lw=0.8)
    ax.set(title="Sales by day of month, compared with a normal day (1.0)", xlabel="day of month")
    save(fig, "03_sales_by_day_of_month.png")
    pay_effect = rel[payday].mean() / rel[~payday].mean() - 1
    notes.append(f"Paydays (15th and last day of the month) sell {pct(pay_effect)} versus other days of the same month"
                 + (": a payday clue is worth building." if pay_effect >= 0.02 else ": no clear payday effect in this data."))

    # ---------------- 4. families ----------------
    fam = df.groupby("family")["sales"].sum().sort_values()
    fig, ax = plt.subplots(figsize=(9, max(4, 0.28 * len(fam))))
    ax.barh(fam.index, fam.values, color="#5b8db8")
    ax.set(title="Total sales by product family", xlabel="units")
    save(fig, "04_sales_by_family.png")
    top3 = fam.sort_values(ascending=False).head(3)
    notes.append(f"{len(fam)} product families; the top 3 ({', '.join(top3.index)}) make "
                 f"{top3.sum() / fam.sum():.0%} of all units, so errors on them matter most.")

    # ---------------- 5. stores ----------------
    st = df.groupby("store_nbr")["sales"].sum().sort_values()
    stype = df.groupby("store_type")["sales"].mean()
    fig, axes = plt.subplots(1, 2, figsize=(13, max(4.5, 0.2 * len(st))), gridspec_kw={"width_ratios": [3, 1]})
    axes[0].barh(st.index.astype(str), st.values, color="#5b8db8")
    axes[0].set(title="Total sales by store", xlabel="units", ylabel="store_nbr")
    axes[1].bar(stype.index, stype.values, color="#e08e45")
    axes[1].set(title="Average sales per row by store type")
    save(fig, "05_sales_by_store.png")
    k = min(10, len(st))
    topk = st.sort_values(ascending=False).head(k).sum() / st.sum()
    notes.append(f"{len(st)} stores; the {k} biggest sell {topk:.0%} of all units. Store type "
                 f"{stype.idxmax()} has the highest average sales per row and type {stype.idxmin()} the lowest.")

    # ---------------- 6. promotions ----------------
    has_promo = df["onpromotion"] > 0
    month = df["date"].dt.to_period("M")
    share = has_promo.groupby(month).mean()
    share.index = share.index.to_timestamp()
    first_promo = df.loc[has_promo, "date"].min()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(share.index, share.values, color="#c0392b")
    axes[0].set(title="Share of rows with a promotion, per month")
    if pd.notna(first_promo):
        after = df["date"] >= first_promo
        means = df.loc[after].groupby(has_promo[after])["sales"].mean()
        axes[1].bar(["no promotion", "promotion"], [means.get(False, np.nan), means.get(True, np.nan)], color=["#5b8db8", "#e08e45"])
        axes[1].set(title=f"Average sales per row (from {first_promo.date()})")
        start_txt = (f"Promotion information starts on {first_promo.date()} (before that onpromotion is always 0). "
                     if first_promo > df["date"].min() else "Promotion information exists from the first day. ")
        notes.append(start_txt + f"Rows with a promotion sell on average {means.get(True, np.nan) / means.get(False, np.nan):.1f} times "
                     "the sales of rows without one (not proof of cause: bigger products are promoted more often).")
    else:
        axes[1].text(0.5, 0.5, "no promotions in the data", ha="center")
        notes.append("onpromotion is always 0 in this data.")
    save(fig, "06_promotions.png")

    # ---------------- 7. holidays and events ----------------
    idx = day.index
    jan1 = (idx.month == 1) & (idx.day == 1)
    cat = np.select(
        [jan1, day["quake"].to_numpy() == 1, day["nat"].to_numpy() == 1, day["ev"].to_numpy() == 1, day["wd"].to_numpy() == 1],
        ["Jan 1", "Earthquake days", "National holiday", "Event day", "Work day"],
        default="Normal day",
    )
    hol = rel.groupby(cat).mean().drop(labels=["Normal day"], errors="ignore")
    order = [c for c in ["National holiday", "Event day", "Work day", "Earthquake days", "Jan 1"] if c in hol.index]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(order, [hol[c] for c in order], color="#5b8db8")
    ax.axhline(1, color="grey", lw=0.8)
    ax.set(title="Daily total vs a normal day of the month (1.0 = normal)")
    save(fig, "07_holidays_events.png")
    sentence = []
    for c in ["National holiday", "Event day", "Work day"]:
        if c in hol.index:
            sentence.append(f"{c.lower()}s {pct(hol[c] - 1)}")
    if sentence:
        notes.append("Compared with a normal day of the same month, " + ", ".join(sentence)
                     + " (first look only: weekday effects are not removed).")
    dec25_present = bool(((idx.month == 12) & (idx.day == 25)).any())
    parts = []
    if "Jan 1" in hol.index:
        parts.append(f"Jan 1 is {pct(hol['Jan 1'] - 1)} versus a normal day" + (" (stores are closed)." if hol["Jan 1"] < 0.2 else "."))
    parts.append("Dec 25 has rows in the data." if dec25_present else "Dec 25 has NO rows at all in any year.")
    notes.append(" ".join(parts))

    # ---------------- 8. earthquake (April 2016) ----------------
    win = total[(total.index >= "2016-03-15") & (total.index <= "2016-05-31")]
    if len(win) > 30 and QUAKE_DAY in total.index:
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(win.index, win.values, color="#9bb7d4", lw=1)
        ax.plot(win.index, win.rolling(7, min_periods=1).mean().values, color="#c0392b", lw=2, label="7-day average")
        ax.axvline(QUAKE_DAY, color="black", ls="--", label="earthquake 2016-04-16")
        ax.set(title="Total sales around the April 2016 earthquake")
        ax.legend()
        save(fig, "08_earthquake_2016.png")
        before = total[(total.index >= QUAKE_DAY - pd.Timedelta(days=14)) & (total.index < QUAKE_DAY)].mean()
        after_q = total[(total.index >= QUAKE_DAY) & (total.index < QUAKE_DAY + pd.Timedelta(days=14))].mean()
        notes.append(f"In the 14 days after the 2016-04-16 earthquake, average daily sales were {pct(after_q / before - 1)} "
                     "compared with the 14 days before.")

    # ---------------- 9. zeros ----------------
    zero_by_fam = (df["sales"] == 0).groupby(df["family"]).mean().sort_values()
    fig, ax = plt.subplots(figsize=(9, max(4, 0.28 * len(zero_by_fam))))
    ax.barh(zero_by_fam.index, zero_by_fam.values, color="#c0392b")
    ax.set(title="Share of days with ZERO sales, by family")
    save(fig, "09_zero_sales_by_family.png")
    n_series = df[["store_nbr", "family"]].drop_duplicates().shape[0]
    dead = df[df["qc_zero_series"] == 1][["store_nbr", "family"]].drop_duplicates().shape[0]
    notes.append(f"{(df['sales'] == 0).mean():.0%} of all rows have zero sales, and {dead} of {n_series} store x family series "
                 f"never sold anything up to {config.CLEANING_STATS_CUTOFF} (see the qc_ flags). Family with the most zero days: "
                 f"{zero_by_fam.idxmax()} ({zero_by_fam.max():.0%}).")

    # ---------------- notes file ----------------
    lines = ["# EDA notes (auto-generated by src/eda.py)", "",
             f"Data: {len(df):,} rows, {df['date'].min().date()} -> {df['date'].max().date()}. "
             "Every number below was computed from the data, not typed by hand.", ""]
    lines += [f"{i}. {n}" for i, n in enumerate(notes, 1)]
    lines += ["", "Charts: see the PNG files in this folder."]
    (config.EDA_DIR / "eda_notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved {len(notes)} notes: {config.EDA_DIR / 'eda_notes.md'}")
    print("\n".join(f"{i}. {n}" for i, n in enumerate(notes, 1)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
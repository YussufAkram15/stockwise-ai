"""StockWise AI dashboard (Level 3, without the chatbot).

Run from the project root:
    streamlit run app/dashboard.py

It only READS files that the scripts already wrote (no training, no raw data):
    reports/forecast_16d.parquet     next 16 days per store x family   (forecast_best_model.py)
    reports/recent_sales.parquet     last 60 real days                 (forecast_best_model.py)
    reports/val_predictions.parquet  the same model on July 2017       (forecast_best_model.py)
    reports/reorder_table.csv        REORDER / OK / OVERSTOCK advice   (inventory.py)
    reports/model_comparison.csv     validation scores of every model  (final_comparison.py)
"""
from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
TODAY = pd.Timestamp("2017-08-15")          # last day of real data
BASELINE_WAPE = 17.50                       # naive lag-21 baseline on validation
STATUS_COLORS = {"REORDER": "#f8d7da", "OK": "#d4edda", "OVERSTOCK": "#ffe5b4"}
HOW_TO_BUILD = ("Run first:  python src/run_pipeline.py  then  python src/forecast_best_model.py  "
                "then  python src/inventory.py")
SIMULATED_NOTE = ("Stock levels and lead times are SIMULATED: the Favorita data has no inventory data. "
                  "The table shows the method, not real orders.")

st.set_page_config(page_title="StockWise AI", page_icon="📦", layout="wide")


# ---------------------------------------------------------------------------
# Loading (cached: each file is read once, not on every click)
# ---------------------------------------------------------------------------
@st.cache_data
def load_parquet(name: str) -> pd.DataFrame | None:
    path = REPORTS / name
    return pd.read_parquet(path) if path.exists() else None


@st.cache_data
def load_csv(name: str, **kwargs) -> pd.DataFrame | None:
    path = REPORTS / name
    return pd.read_csv(path, **kwargs) if path.exists() else None


forecast = load_parquet("forecast_16d.parquet")
recent = load_parquet("recent_sales.parquet")
val = load_parquet("val_predictions.parquet")
reorder = load_csv("reorder_table.csv", parse_dates=["stockout_date"])
comparison = load_csv("model_comparison.csv")

if forecast is None or recent is None or reorder is None:
    st.error("The forecast or the reorder table is missing. " + HOW_TO_BUILD)
    st.stop()

MODEL = forecast["model"].iloc[0]
STORES = sorted(reorder["store_nbr"].unique())
FAMILIES = sorted(reorder["family"].unique())


def color_status(value: str) -> str:
    return f"background-color: {STATUS_COLORS.get(value, '')}; color: #1a1a1a"


def show_table(table: pd.DataFrame) -> None:
    """Show the reorder rows with the status column colored."""
    shown = table.copy()
    shown["stockout_date"] = shown["stockout_date"].dt.date
    styled = shown.style.map(color_status, subset=["status"]).format(precision=1, thousands=",")
    st.dataframe(styled, hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("📦 StockWise AI")
PAGES = ["Overview", "Forecast", "Reorder table", "Model accuracy"]
# A page can also be opened by link, e.g. http://localhost:8501/?page=Forecast
start = st.query_params.get("page", "Overview")
page = st.sidebar.radio("Page", PAGES, index=PAGES.index(start) if start in PAGES else 0)
st.sidebar.markdown(f"**Today:** {TODAY.date()} (last day of data)  \n**Forecast model:** {MODEL}  \n"
                    f"**Horizon:** {forecast['date'].min().date()} → {forecast['date'].max().date()}")

# ---------------------------------------------------------------------------
# Page 1: Overview
# ---------------------------------------------------------------------------
if page == "Overview":
    st.title("Overview")
    st.info(SIMULATED_NOTE)

    counts = reorder["status"].value_counts().reindex(["REORDER", "OK", "OVERSTOCK"], fill_value=0)
    best_wape = None
    if comparison is not None:
        best_wape = comparison.loc[comparison["model"] == MODEL, "WAPE_%"]
        best_wape = float(best_wape.iloc[0]) if len(best_wape) else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Store × family series", f"{len(reorder):,}")
    c2.metric("Need to REORDER", f"{counts['REORDER']:,}")
    c3.metric("OVERSTOCK", f"{counts['OVERSTOCK']:,}")
    if best_wape is not None:
        c4.metric(f"{MODEL} validation WAPE", f"{best_wape:.2f}%",
                  delta=f"{best_wape - BASELINE_WAPE:.2f} pts vs baseline", delta_color="inverse")
    else:
        c4.metric("Baseline validation WAPE", f"{BASELINE_WAPE:.2f}%")

    st.subheader("Advice for every series")
    bars = alt.Chart(counts.rename_axis("status").reset_index(name="series")).mark_bar().encode(
        x=alt.X("status:N", sort=["REORDER", "OK", "OVERSTOCK"], title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("series:Q", title="series"),
        color=alt.Color("status:N", legend=None, scale=alt.Scale(
            domain=["REORDER", "OK", "OVERSTOCK"], range=["#d62728", "#2ca02c", "#ff7f0e"])),
        tooltip=["status:N", "series:Q"],
    )
    st.altair_chart(bars.properties(height=300), width="stretch")

    st.subheader("The 10 most urgent items")
    urgent = (reorder[reorder["status"] == "REORDER"]
              .sort_values(["stockout_date", "order_qty"], ascending=[True, False]).head(10))
    show_table(urgent[["store_nbr", "family", "on_hand", "reorder_point", "stockout_date", "order_qty"]]
               .assign(status="REORDER"))

    st.subheader("Total forecast for all stores, per day")
    st.line_chart(forecast.groupby("date")["forecast"].sum().rename("forecast (units)"))

# ---------------------------------------------------------------------------
# Page 2: Forecast for one store and family
# ---------------------------------------------------------------------------
elif page == "Forecast":
    st.title("Forecast")
    c1, c2 = st.columns(2)
    store = c1.selectbox("Store", STORES)
    family = c2.selectbox("Family", FAMILIES, index=FAMILIES.index("GROCERY I") if "GROCERY I" in FAMILIES else 0)

    def pick(df: pd.DataFrame) -> pd.DataFrame:
        return df[(df["store_nbr"] == store) & (df["family"] == family)]

    actual = pick(recent)[["date", "sales"]].rename(columns={"sales": "units"}).assign(line="actual sales")
    future = pick(forecast)[["date", "forecast"]].rename(columns={"forecast": "units"}).assign(line="forecast")
    both = pd.concat([actual, future])

    lines = alt.Chart(both).mark_line(point=True).encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("units:Q", title="units sold"),
        color=alt.Color("line:N", title=None, scale=alt.Scale(
            domain=["actual sales", "forecast"], range=["#4c78a8", "#f58518"])),
        strokeDash=alt.StrokeDash("line:N", legend=None),
        tooltip=["date:T", "line:N", alt.Tooltip("units:Q", format=",.1f")],
    )
    today = alt.Chart(pd.DataFrame({"date": [TODAY]})).mark_rule(color="grey", strokeDash=[4, 4]).encode(x="date:T")
    st.altair_chart((lines + today).properties(height=380), width="stretch")
    st.caption("Grey dashed line = today (2017-08-15). Left of it: real sales. Right of it: the model's forecast.")

    row = pick(reorder)
    if len(row):
        r = row.iloc[0]
        st.subheader(f"Stock advice: {r['status']}")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("On hand (simulated)", f"{r['on_hand']:,.1f}")
        c2.metric("Reorder point", f"{r['reorder_point']:,.1f}")
        c3.metric("Safety stock", f"{r['safety_stock']:,.1f}")
        c4.metric("Order quantity", f"{r['order_qty']:,.0f}")
        c5.metric("Runs out on", "—" if pd.isna(r["stockout_date"]) else str(r["stockout_date"].date()))
        st.caption(f"\u2014 = does not run out within the 16 days · "
                   f"Lead time {r['lead_time_days']:.0f} days (simulated) · average forecast "
                   f"{r['avg_daily_forecast']:,.1f} units/day · usual forecast error (σ) {r['sigma']:,.1f}")

# ---------------------------------------------------------------------------
# Page 3: Reorder table with filters
# ---------------------------------------------------------------------------
elif page == "Reorder table":
    st.title("Reorder table")
    st.info(SIMULATED_NOTE)

    c1, c2, c3, c4 = st.columns(4)
    statuses = c1.multiselect("Status", ["REORDER", "OK", "OVERSTOCK"], default=["REORDER"])
    stores = c2.multiselect("Store", STORES)
    families = c3.multiselect("Family", FAMILIES)
    before = c4.date_input("Runs out before", value=None,
                           min_value=forecast["date"].min().date(), max_value=forecast["date"].max().date())

    table = reorder[reorder["status"].isin(statuses)]
    if stores:
        table = table[table["store_nbr"].isin(stores)]
    if families:
        table = table[table["family"].isin(families)]
    if before:
        table = table[table["stockout_date"] < pd.Timestamp(before)]

    st.write(f"**{len(table):,} rows**")
    show_table(table)
    st.download_button("Download as CSV", table.to_csv(index=False), "reorder_selection.csv", "text/csv")
    st.caption("Example: Status = REORDER, Family = BEVERAGES, Runs out before = 2017-08-23 "
               "answers \"which stores may run out of BEVERAGES next week?\"")

# ---------------------------------------------------------------------------
# Page 4: Model accuracy
# ---------------------------------------------------------------------------
else:
    st.title("Model accuracy (validation, July 2017)")
    st.write(f"A model is only useful if its validation WAPE is lower than the naive baseline "
             f"(**{BASELINE_WAPE:.2f}%**: copy the sales of 21 days earlier). Lower is better.")

    if comparison is None:
        st.info("Model comparison not found. Run: python src/final_comparison.py")
    else:
        scores = comparison[comparison["kind"] != "experiment"][["model", "MAE", "RMSE", "WAPE_%", "better_than_naive_%"]]
        st.dataframe(scores.round(2), hide_index=True, width="stretch")
        bars = alt.Chart(scores).mark_bar().encode(
            x=alt.X("WAPE_%:Q", title="validation WAPE % (lower is better)"),
            y=alt.Y("model:N", sort="x", title=None),
            color=alt.condition(alt.datum.model == MODEL, alt.value("#f58518"), alt.value("#4c78a8")),
            tooltip=["model:N", alt.Tooltip("WAPE_%:Q", format=".2f")],
        )
        rule = alt.Chart(pd.DataFrame({"x": [BASELINE_WAPE]})).mark_rule(color="grey", strokeDash=[4, 4]).encode(x="x:Q")
        st.altair_chart((bars + rule).properties(height=220), width="stretch")

    if val is not None:
        st.subheader(f"{MODEL}: real vs predicted in July for one series")
        c1, c2 = st.columns(2)
        store = c1.selectbox("Store", STORES, key="acc_store")
        family = c2.selectbox("Family", FAMILIES, key="acc_family",
                              index=FAMILIES.index("GROCERY I") if "GROCERY I" in FAMILIES else 0)
        one = val[(val["store_nbr"] == store) & (val["family"] == family)].set_index("date")
        st.line_chart(one[["sales", "prediction"]], color=["#4c78a8", "#f58518"])

"""LEVEL-1 COMPARISON: one fair table for all models + residual analysis + charts.

Run from the project root (after the models were trained):
    python src/final_comparison.py

It reads the validation predictions of every model that exists
    reports/model_comparison_results.parquet     (Linear Regression, Random Forest, XGBoost)
    reports/polynomial_val_predictions*.parquet  (Polynomial Regression and its experiments)
adds the NAIVE BASELINE ("same as 21 days ago", computed here from the real sales),
and scores everything with the SAME code on the SAME rows (MAE, RMSE, WAPE).

Writes   reports/model_comparison.md   table + plain-language residual analysis
         reports/model_comparison.csv
         reports/figures/*.png
It never touches the test slice (31 Jul - 15 Aug): only validation rows are used.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("matplotlib is missing. Run:  pip install matplotlib")

import config
from data_io import setup_console

KEY = ["store_nbr", "family"]
FIG_DIR = config.REPORTS_DIR / "figures"
BASELINE_NAME = "NaiveBaseline (same as 21 days ago)"
REQUIRED = ["id", "date", "store_nbr", "family", "actual_sales", "model", "predicted_sales"]
FEATURES_PATH = getattr(config, "FEATURES_TRAIN_PATH", config.PROCESSED_DIR / "train_features.parquet")


# ----------------------------------------------------------------------------- scores
def score(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    pred = np.clip(pred, 0, None)
    total = float(np.sum(actual))
    return {
        "MAE": float(np.mean(np.abs(actual - pred))),
        "RMSE": float(np.sqrt(np.mean((actual - pred) ** 2))),
        "WAPE_%": float("nan") if total <= 0 else 100.0 * float(np.sum(np.abs(actual - pred))) / total,
    }


def kind_of(name: str) -> str:
    if name.startswith("NaiveBaseline"):
        return "baseline"
    if name.startswith("PolynomialRegression_"):
        return "experiment"          # --more-clues / --log-target / --exclude-outliers
    return "model"


# ----------------------------------------------------------------------------- data
def load_predictions() -> pd.DataFrame:
    frames, sources = [], []
    p = config.REPORTS_DIR / "model_comparison_results.parquet"
    if p.exists():
        frames.append(pd.read_parquet(p)); sources.append(p.name)
    for q in sorted(config.REPORTS_DIR.glob("polynomial_val_predictions*.parquet")):
        frames.append(pd.read_parquet(q)); sources.append(q.name)
    if not frames:
        raise SystemExit("No prediction files found in reports/. Train the models first.")
    out = pd.concat(frames, ignore_index=True)
    missing = [c for c in REQUIRED if c not in out.columns]
    if missing:
        raise SystemExit(f"Prediction files are missing columns: {missing}")
    out["date"] = pd.to_datetime(out["date"])
    print("Prediction files used: " + ", ".join(sources))
    return out


def naive_baseline(ids_frame: pd.DataFrame) -> pd.DataFrame:
    """Predict the sales of the same store x family 21 days earlier (real `sales`, never the capped ones)."""
    if not FEATURES_PATH.exists():
        raise SystemExit(f"{FEATURES_PATH} not found. Run feature_engineering.py first.")
    hist = pd.read_parquet(FEATURES_PATH, columns=KEY + ["date", "sales"])
    hist["date"] = pd.to_datetime(hist["date"])
    hist = hist.rename(columns={"sales": "predicted_sales"})
    hist["date"] = hist["date"] + pd.Timedelta(days=21)          # sales of day d-21 becomes the prediction for day d
    base = ids_frame.merge(hist, on=KEY + ["date"], how="left")
    base["predicted_sales"] = base["predicted_sales"].fillna(0.0)
    base["model"] = BASELINE_NAME
    return base[REQUIRED]


def main() -> int:
    setup_console()
    preds = load_predictions()

    # same rows for every model (otherwise the comparison is not fair)
    ids = None
    for _, g in preds.groupby("model"):
        ids = set(g["id"]) if ids is None else ids & set(g["id"])
    sizes = preds.groupby("model")["id"].nunique()
    if sizes.nunique() > 1:
        print(f"NOTE: models have different numbers of rows ({sizes.to_dict()}); using the {len(ids):,} rows they all have.")
    preds = preds[preds["id"].isin(ids)]
    if preds["date"].max() >= pd.Timestamp("2017-07-31"):
        raise SystemExit("Predictions reach the TEST slice (31 Jul or later). Fix VAL_END = 2017-07-30 and retrain.")
    if preds[["actual_sales", "predicted_sales"]].isna().any().any():
        raise SystemExit("Missing values in the predictions.")

    one = preds.drop_duplicates("id")[["id", "date", "store_nbr", "family", "actual_sales"]]
    base = naive_baseline(one[["id", "date", "store_nbr", "family", "actual_sales"]])
    allp = pd.concat([preds[REQUIRED], base], ignore_index=True)

    # ---- table -----------------------------------------------------------------
    rows = []
    for name, g in allp.groupby("model"):
        s = score(g["actual_sales"].to_numpy(), g["predicted_sales"].to_numpy())
        rows.append({"model": name, "kind": kind_of(name), "rows": len(g), **s})
    table = pd.DataFrame(rows).sort_values("WAPE_%").reset_index(drop=True)
    naive_wape = float(table.loc[table["kind"] == "baseline", "WAPE_%"].iloc[0])
    table["better_than_naive_%"] = (100 * (naive_wape - table["WAPE_%"]) / naive_wape).round(1)
    table.to_csv(config.REPORTS_DIR / "model_comparison.csv", index=False)

    official = table[table["kind"] == "model"]
    best_name = (official if len(official) else table[table["kind"] != "baseline"]).iloc[0]["model"]
    print("\n" + table.round(3).to_string(index=False))
    print(f"\nBest official model (lowest WAPE): {best_name}")

    # ---- residual analysis of the best model ------------------------------------
    d = allp[allp["model"] == best_name].copy()
    d["pred"] = d["predicted_sales"].clip(lower=0)
    d["err"] = (d["actual_sales"] - d["pred"]).abs()
    total_err = float(d["err"].sum())
    available = _cols()
    flags = [c for c in ("is_holiday", "qc_store_closed_day") if c in available]
    extra = pd.read_parquet(FEATURES_PATH, columns=["id"] + flags) if flags else d[["id"]]
    d = d.merge(extra, on="id", how="left")

    bias = float(d["pred"].sum() / d["actual_sales"].sum())
    fam = d.groupby("family").agg(err=("err", "sum"), act=("actual_sales", "sum"))
    fam["share_%"] = 100 * fam["err"] / total_err
    fam["wape_%"] = 100 * fam["err"] / fam["act"].where(fam["act"] > 0)
    sto = d.groupby("store_nbr")["err"].sum().sort_values(ascending=False)
    day = d.groupby("date").agg(err=("err", "sum"), act=("actual_sales", "sum"), pred=("pred", "sum"))
    day["wape_%"] = 100 * day["err"] / day["act"].where(day["act"] > 0)
    ser = d.groupby(KEY).agg(err=("err", "sum"), act=("actual_sales", "sum"))
    ser["bucket"] = pd.qcut(ser["act"].rank(method="first"), 4, labels=["smallest 25%", "25-50%", "50-75%", "biggest 25%"])
    by_size = ser.groupby("bucket", observed=True).agg(err=("err", "sum"), act=("act", "sum"))
    by_size["share_%"] = 100 * by_size["err"] / total_err
    by_size["wape_%"] = 100 * by_size["err"] / by_size["act"].where(by_size["act"] > 0)
    worst = d.sort_values("err", ascending=False).head(10)[["date", "store_nbr", "family", "actual_sales", "pred", "err"]]
    top1 = float(d["err"].nlargest(max(1, len(d) // 100)).sum() / total_err * 100)

    # ---- charts -------------------------------------------------------------------
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(table) + 2))
    colors = ["#c0392b" if k == "baseline" else ("#9bb7d4" if k == "experiment" else "#5b8db8") for k in table["kind"]]
    ax.barh(table["model"][::-1], table["WAPE_%"][::-1], color=colors[::-1])
    ax.axvline(naive_wape, color="black", ls="--", lw=1, label=f"naive baseline {naive_wape:.2f}%")
    ax.set(title="Validation WAPE by model (lower is better)", xlabel="WAPE %")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "model_wape_comparison.png", dpi=110); plt.close(fig)

    s = d.sample(min(20000, len(d)), random_state=0)
    hi = float(max(s["actual_sales"].max(), s["pred"].max(), 1))
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(s["actual_sales"] + 1, s["pred"] + 1, s=4, alpha=0.3, color="#5b8db8")
    ax.plot([1, hi + 1], [1, hi + 1], color="#c0392b", lw=1)
    ax.set(xscale="log", yscale="log", xlabel="actual sales + 1", ylabel="predicted sales + 1",
           title=f"Predicted vs actual ({best_name})")
    fig.tight_layout(); fig.savefig(FIG_DIR / "predicted_vs_actual.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(day.index, day["act"], label="actual", color="#2c3e50")
    ax.plot(day.index, day["pred"], label="predicted", color="#e08e45")
    ax.set(title=f"Total sales per day, validation ({best_name})", ylabel="units")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "daily_actual_vs_predicted.png", dpi=110); plt.close(fig)

    top_fam = fam.sort_values("share_%", ascending=False).head(15).sort_values("share_%")
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(top_fam) + 2))
    ax.barh(top_fam.index, top_fam["share_%"], color="#5b8db8")
    ax.set(title="Share of the total error by product family (top 15)", xlabel="% of total absolute error")
    fig.tight_layout(); fig.savefig(FIG_DIR / "error_share_by_family.png", dpi=110); plt.close(fig)

    # ---- markdown report ----------------------------------------------------------
    L: list[str] = []
    a = L.append
    a("# Level 1 - model comparison (auto-generated by src/final_comparison.py)")
    a("")
    a(f"Validation: {d['date'].min().date()} -> {d['date'].max().date()}, **{len(d):,} rows, the same rows for every model.** "
      "The test slice (31 Jul - 15 Aug 2017) is NOT used here.")
    a("Scores use the same code for all models; predictions are clipped to 0. WAPE is the main metric.")
    a("")
    a("| model | kind | MAE | RMSE | WAPE % | better than naive % |")
    a("|---|---|---|---|---|---|")
    for _, r in table.iterrows():
        a(f"| {r['model']} | {r['kind']} | {r['MAE']:.2f} | {r['RMSE']:.2f} | {r['WAPE_%']:.2f} | {r['better_than_naive_%']:+.1f} |")
    a("")
    a("`experiment` rows are extra runs of the Polynomial model (not part of the agreed 5 models). "
      "`better than naive` = how much lower the WAPE is than the baseline.")
    a("")
    a(f"## Residual analysis of the best official model: {best_name}")
    a("")
    a(f"- **Bias:** total predicted / total actual = {bias:.3f} "
      + ("(it predicts too little)." if bias < 0.98 else "(it predicts too much)." if bias > 1.02 else "(no clear over- or under-prediction)."))
    a(f"- **Concentration:** the worst 1% of rows cause {top1:.0f}% of the total error.")
    a("- **By series size** (series sorted by total sales, in 4 equal groups):")
    for k, r in by_size.iterrows():
        a(f"  - {k}: {r['share_%']:.0f}% of the error, WAPE {r['wape_%']:.1f}%")
    a("- **Product families** with the largest share of the error: "
      + ", ".join(f"{i} ({r['share_%']:.0f}%, WAPE {r['wape_%']:.0f}%)" for i, r in fam.sort_values("share_%", ascending=False).head(5).iterrows()) + ".")
    a("- **Stores** with the largest share of the error: "
      + ", ".join(f"store {i} ({100 * v / total_err:.0f}%)" for i, v in sto.head(5).items()) + ".")
    wd = day["wape_%"].dropna()
    if len(wd):
        a(f"- **Over time:** the best day has WAPE {wd.min():.1f}% ({wd.idxmin().date()}), the worst {wd.max():.1f}% ({wd.idxmax().date()}).")
    if "is_holiday" in d.columns:
        for flag, label in ((0, "normal days"), (1, "holidays")):
            g = d[d["is_holiday"].fillna(0) == flag]
            if len(g) and g["actual_sales"].sum() > 0:
                a(f"- **{label.capitalize()}:** WAPE {100 * g['err'].sum() / g['actual_sales'].sum():.1f}% on {len(g):,} rows.")
    if "qc_store_closed_day" in d.columns and (d["qc_store_closed_day"].fillna(0) == 1).any():
        g = d[d["qc_store_closed_day"] == 1]
        a(f"- **Store-closed days:** {len(g):,} rows ({100 * len(g) / len(d):.1f}%) cause {100 * g['err'].sum() / total_err:.0f}% of the error.")
    a("")
    a("### The 10 biggest single errors")
    a("")
    a("| date | store | family | actual | predicted | error |")
    a("|---|---|---|---|---|---|")
    for _, r in worst.iterrows():
        a(f"| {r['date'].date()} | {int(r['store_nbr'])} | {r['family']} | {r['actual_sales']:,.0f} | {r['pred']:,.0f} | {r['err']:,.0f} |")
    a("")
    a("Charts: `reports/figures/model_wape_comparison.png`, `predicted_vs_actual.png`, "
      "`daily_actual_vs_predicted.png`, `error_share_by_family.png`.")
    (config.REPORTS_DIR / "model_comparison.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nSaved: {config.REPORTS_DIR / 'model_comparison.md'}   (+ model_comparison.csv, figures/)")
    return 0


def _cols() -> list[str]:
    import pyarrow.parquet as pq
    return pq.read_schema(FEATURES_PATH).names


if __name__ == "__main__":
    raise SystemExit(main())
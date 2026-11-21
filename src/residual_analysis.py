"""RESIDUAL ANALYSIS: where, when and why is the model wrong?

Run from the project root (after train_models.py / train_polynomial.py):
    python src/residual_analysis.py                       # the best official model
    python src/residual_analysis.py --model RandomForest  # any model that has validation predictions
    python src/residual_analysis.py --shap-rows 5000      # more rows for SHAP (default 3000)
    python src/residual_analysis.py --no-shap             # skip SHAP

A residual = actual sales - predicted sales  (positive = the model predicted too little).
Only VALIDATION rows are used. The test slice (31 Jul - 15 Aug) is never touched.

Writes  reports/residual_analysis.md
        reports/figures/res_01_actual_vs_predicted.png      1. Actual vs Predicted
        reports/figures/res_02_residual_distribution.png    2. Residual distribution
        reports/figures/res_03_residuals_over_time.png      3. Residuals over time
        reports/figures/res_04_error_by_store.png           4. Error by store
        reports/figures/res_05_error_by_family.png          5. Error by family
        reports/figures/res_06_holidays_vs_normal.png       6. Error on holidays vs normal days
        reports/figures/res_07_feature_importance.png       7. Feature importance
        reports/figures/res_08_shap_summary.png             7. SHAP (what pushes each prediction up or down)
        reports/figures/res_09_shap_bar.png                 7. SHAP (average size of each clue)
"""
from __future__ import annotations

import argparse
import json
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

FIG_DIR = getattr(config, "FIGURES_DIR", config.REPORTS_DIR / "figures")
FEATURES_PATH = getattr(config, "FEATURES_TRAIN_PATH", config.PROCESSED_DIR / "train_features.parquet")
BEST_MODEL_PATH = getattr(config, "BEST_MODEL_PATH", config.PROJECT_ROOT / "models" / "best_model.pkl")
METADATA_PATH = getattr(config, "MODEL_COMPARISON_METADATA_PATH", config.REPORTS_DIR / "model_comparison_metadata.json")
RESULTS_PATH = getattr(config, "MODEL_COMPARISON_RESULTS_PATH", config.REPORTS_DIR / "model_comparison_results.parquet")
KEY = ["store_nbr", "family"]
BLUE, ORANGE, RED, DARK = "#5b8db8", "#e08e45", "#c0392b", "#2c3e50"
FLAG_COLS = ["is_holiday", "is_holiday_national", "is_holiday_regional", "is_holiday_local", "is_event",
             "is_payday", "is_weekend", "qc_store_closed_day"]


def wape(err: float, actual: float) -> float:
    return float("nan") if actual <= 0 else 100.0 * err / actual


def save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=110)
    plt.close(fig)
    print(f"  saved {name}")


# ----------------------------------------------------------------------------- data
def load_predictions(model: str | None) -> tuple[pd.DataFrame, str]:
    frames = []
    if RESULTS_PATH.exists():
        frames.append(pd.read_parquet(RESULTS_PATH))
    for p in sorted(config.REPORTS_DIR.glob("polynomial_val_predictions*.parquet")):
        frames.append(pd.read_parquet(p))
    if not frames:
        raise SystemExit("No predictions found in reports/. Train the models first.")
    allp = pd.concat(frames, ignore_index=True)
    if model is None:
        if METADATA_PATH.exists():
            model = json.loads(METADATA_PATH.read_text(encoding="utf-8")).get("best_model")
        model = model or allp["model"].iloc[0]
    d = allp[allp["model"] == model].copy()
    if d.empty:
        raise SystemExit(f"No predictions for '{model}'. Available: {sorted(allp['model'].unique())}")
    d["date"] = pd.to_datetime(d["date"])
    if d["date"].max() >= pd.Timestamp(getattr(config, "TEST_START", "2017-07-31")):
        raise SystemExit("Predictions reach the TEST slice. Fix VAL_END and retrain.")
    d["pred"] = d["predicted_sales"].clip(lower=0)
    d["resid"] = d["actual_sales"] - d["pred"]            # > 0: predicted too little
    d["err"] = d["resid"].abs()
    return d, model


def add_flags(d: pd.DataFrame) -> pd.DataFrame:
    import pyarrow.parquet as pq
    if not FEATURES_PATH.exists():
        return d
    have = [c for c in FLAG_COLS if c in set(pq.read_schema(FEATURES_PATH).names)]
    if not have:
        return d
    return d.merge(pd.read_parquet(FEATURES_PATH, columns=["id"] + have), on="id", how="left")


# ----------------------------------------------------------------------------- 1-6
def plot_actual_vs_predicted(d: pd.DataFrame, model: str) -> dict:
    s = d.sample(min(20000, len(d)), random_state=0)
    hi = float(max(s["actual_sales"].max(), s["pred"].max(), 1))
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    axes[0].scatter(s["actual_sales"] + 1, s["pred"] + 1, s=4, alpha=0.25, color=BLUE)
    axes[0].plot([1, hi + 1], [1, hi + 1], color=RED, lw=1, label="perfect prediction")
    axes[0].set(xscale="log", yscale="log", xlabel="actual sales + 1", ylabel="predicted sales + 1",
                title="Every point is one store x family x day (log scale)")
    axes[0].legend()
    # average prediction inside 20 groups of similar actual sales: is the model biased for small / large sales?
    pos = d[d["actual_sales"] > 0].copy()
    pos["bin"] = pd.qcut(pos["actual_sales"].rank(method="first"), 20, labels=False)
    g = pos.groupby("bin").agg(actual=("actual_sales", "mean"), pred=("pred", "mean"))
    axes[1].plot(g["actual"], g["pred"], "o-", color=ORANGE)
    axes[1].plot([0, g["actual"].max()], [0, g["actual"].max()], color=RED, lw=1)
    axes[1].set(xscale="symlog", yscale="symlog", xlabel="average actual sales of the group",
                ylabel="average prediction of the group", title="Average prediction vs average actual (20 equal groups)")
    fig.suptitle(f"Actual vs Predicted - {model}")
    save(fig, "res_01_actual_vs_predicted.png")
    big = g.iloc[-1]
    small = g.iloc[0]
    return {"small_ratio": float(small["pred"] / small["actual"]) if small["actual"] > 0 else float("nan"),
            "big_ratio": float(big["pred"] / big["actual"])}


def plot_residual_distribution(d: pd.DataFrame, model: str) -> dict:
    lo, hi = d["resid"].quantile([0.01, 0.99])
    log_err = np.log1p(d["pred"]) - np.log1p(d["actual_sales"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(d["resid"].clip(lo, hi), bins=100, color=BLUE)
    axes[0].axvline(0, color=RED, ls="--")
    axes[0].set(title="Residual = actual - predicted (units, 1% - 99% of the rows)", xlabel="units")
    axes[1].hist(log_err.clip(-3, 3), bins=100, color=ORANGE)
    axes[1].axvline(0, color=RED, ls="--")
    axes[1].set(title="Log error = log(1+predicted) - log(1+actual)", xlabel="log error (0 = perfect, > 0 = too high)")
    fig.suptitle(f"Residual distribution - {model}")
    save(fig, "res_02_residual_distribution.png")
    return {"mean": float(d["resid"].mean()), "median": float(d["resid"].median()), "std": float(d["resid"].std()),
            "under_%": float(100 * (d["resid"] > 0).mean()), "over_%": float(100 * (d["resid"] < 0).mean()),
            "exact_%": float(100 * (d["resid"] == 0).mean()), "skew": float(d["resid"].skew())}


def plot_residuals_over_time(d: pd.DataFrame, model: str) -> pd.DataFrame:
    day = d.groupby("date").agg(resid=("resid", "sum"), err=("err", "sum"), act=("actual_sales", "sum"), pred=("pred", "sum"))
    day["wape"] = 100 * day["err"] / day["act"].where(day["act"] > 0)
    day["bias_%"] = 100 * day["resid"] / day["act"].where(day["act"] > 0)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].bar(day.index, day["resid"], color=[RED if v > 0 else BLUE for v in day["resid"]])
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set(title="Total residual per day (red = predicted too little, blue = too much)", ylabel="units")
    axes[1].plot(day.index, day["wape"], "o-", color=DARK, label="WAPE of the day")
    axes[1].plot(day.index, day["bias_%"], "s--", color=ORANGE, label="bias % (actual - predicted) / actual")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set(ylabel="%")
    axes[1].legend()
    fig.suptitle(f"Residuals over time - {model}")
    save(fig, "res_03_residuals_over_time.png")
    return day


def group_table(d: pd.DataFrame, by: str) -> pd.DataFrame:
    g = d.groupby(by).agg(err=("err", "sum"), act=("actual_sales", "sum"), resid=("resid", "sum"), rows=("err", "size"))
    g["share_%"] = 100 * g["err"] / g["err"].sum()
    g["wape_%"] = 100 * g["err"] / g["act"].where(g["act"] > 0)
    g["bias_%"] = 100 * g["resid"] / g["act"].where(g["act"] > 0)
    return g


def plot_error_by(d: pd.DataFrame, by: str, title: str, fname: str, top: int = 15) -> pd.DataFrame:
    g = group_table(d, by)
    t = g.sort_values("share_%", ascending=False).head(top).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(13, max(4.5, 0.38 * len(t) + 1.5)), sharey=True)
    axes[0].barh(t.index.astype(str), t["share_%"], color=BLUE)
    axes[0].set(title=f"Share of the total error (top {len(t)})", xlabel="% of the total absolute error")
    axes[1].barh(t.index.astype(str), t["wape_%"], color=ORANGE)
    axes[1].axvline(100 * d["err"].sum() / d["actual_sales"].sum(), color=RED, ls="--", label="overall WAPE")
    axes[1].set(title="WAPE of each (lower is better)", xlabel="WAPE %")
    axes[1].legend()
    fig.suptitle(title)
    save(fig, fname)
    return g


def holiday_table(d: pd.DataFrame) -> pd.DataFrame:
    cats: dict[str, pd.Series] = {}
    hol = d["is_holiday"].fillna(0) == 1 if "is_holiday" in d else pd.Series(False, index=d.index)
    ev = d["is_event"].fillna(0) == 1 if "is_event" in d else pd.Series(False, index=d.index)
    cats["normal days (no holiday, no event)"] = ~hol & ~ev
    cats["holidays (any scope)"] = hol
    if "is_holiday_national" in d:
        cats["  national holidays"] = d["is_holiday_national"].fillna(0) == 1
    if "is_holiday_local" in d:
        cats["  local holidays"] = d["is_holiday_local"].fillna(0) == 1
    cats["events (Black Friday, Mother's Day ...)"] = ev
    if "is_payday" in d:
        cats["paydays (15th, month end)"] = d["is_payday"].fillna(0) == 1
    if "is_weekend" in d:
        cats["weekends"] = d["is_weekend"].fillna(0) == 1
        cats["weekdays"] = d["is_weekend"].fillna(0) == 0
    if "qc_store_closed_day" in d:
        cats["store closed (sold nothing)"] = d["qc_store_closed_day"].fillna(0) == 1
    rows = []
    for name, mask in cats.items():
        g = d[mask]
        if len(g) == 0:
            continue
        rows.append({"group": name, "rows": len(g), "days": g["date"].nunique(), "wape_%": wape(g["err"].sum(), g["actual_sales"].sum()),
                     "bias_%": 100 * g["resid"].sum() / g["actual_sales"].sum() if g["actual_sales"].sum() > 0 else float("nan"),
                     "share_of_error_%": 100 * g["err"].sum() / d["err"].sum()})
    return pd.DataFrame(rows)


def plot_holidays(d: pd.DataFrame, model: str) -> pd.DataFrame:
    t = holiday_table(d)
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.45 * len(t) + 2.5), sharey=True)
    y = t["group"].str.strip()[::-1]
    axes[0].barh(y, t["wape_%"][::-1], color=BLUE)
    axes[0].axvline(100 * d["err"].sum() / d["actual_sales"].sum(), color=RED, ls="--", label="overall WAPE")
    axes[0].set(title="WAPE", xlabel="%")
    axes[0].legend()
    axes[1].barh(y, t["bias_%"][::-1], color=[RED if v > 0 else ORANGE for v in t["bias_%"][::-1]])
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set(title="Bias (> 0: predicted too little)", xlabel="%")
    fig.suptitle(f"Error on holidays vs normal days - {model}")
    save(fig, "res_06_holidays_vs_normal.png")
    return t


# ----------------------------------------------------------------------------- 7. importance + SHAP
def plot_importance(model: str) -> dict | None:
    if not METADATA_PATH.exists():
        return None
    meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    entry = next((m for m in meta["models"] if m["name"] == model), None)
    imp = (entry or {}).get("importances") or {}
    if not imp:
        print("  feature importance skipped: no importances saved for this model")
        return None
    top = sorted(imp.items(), key=lambda kv: -kv[1])[:20][::-1]
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(top) + 2))
    ax.barh([k for k, _ in top], [v for _, v in top], color=BLUE)
    ax.set(title=f"Top 20 feature importances - {model}", xlabel="importance (model-specific scale)")
    save(fig, "res_07_feature_importance.png")
    return dict(sorted(imp.items(), key=lambda kv: -kv[1])[:10])


def run_shap(model: str, d: pd.DataFrame, rows: int) -> dict | None:
    try:
        import shap
        import joblib
    except ImportError:
        print("  SHAP skipped: run  pip install shap  to get it")
        return None
    if not BEST_MODEL_PATH.exists():
        print("  SHAP skipped: models/best_model.pkl not found (run train_models.py)")
        return None
    pack = joblib.load(BEST_MODEL_PATH)
    if pack.get("name") != model:
        print(f"  SHAP skipped: the saved model is '{pack.get('name')}', not '{model}'")
        return None
    feats = pack["features"]
    obj, transform = pack["model"], pack.get("target_transform", "none")

    ids = d["id"].sample(min(rows, len(d)), random_state=0)
    X = pd.read_parquet(FEATURES_PATH, columns=["id"] + feats)
    X = X[X["id"].isin(set(ids))][feats].astype("float64")

    if isinstance(obj, tuple):                                   # XGBoost: (imputer, model)
        imputer, inner = obj
        Xi = pd.DataFrame(imputer.transform(X.to_numpy(dtype="float32")), columns=feats)
        explainer = shap.TreeExplainer(inner)
        values = explainer.shap_values(Xi, check_additivity=False)
    elif hasattr(obj, "named_steps"):
        steps = obj.named_steps
        inner = steps["model"]
        Xi = pd.DataFrame(steps["imputer"].transform(X.to_numpy(dtype="float32")), columns=feats)
        if "scaler" in steps:                                    # Linear Regression
            Xs = steps["scaler"].transform(Xi.to_numpy())
            values = shap.LinearExplainer(inner, Xs).shap_values(Xs)
            Xi = pd.DataFrame(Xs, columns=feats)
        else:                                                    # Random Forest
            values = shap.TreeExplainer(inner).shap_values(Xi, check_additivity=False)
    else:
        print("  SHAP skipped: this model type is not supported")
        return None

    unit = "log(1 + sales)" if transform == "log1p" else "units of sales"
    plt.figure()
    shap.summary_plot(values, Xi, max_display=20, show=False)
    plt.title(f"SHAP summary - {model}  (effect on the prediction, in {unit})")
    save(plt.gcf(), "res_08_shap_summary.png")
    plt.figure()
    shap.summary_plot(values, Xi, plot_type="bar", max_display=20, show=False)
    plt.title(f"SHAP: average size of each clue - {model}  ({unit})")
    save(plt.gcf(), "res_09_shap_bar.png")
    mean_abs = pd.Series(np.abs(values).mean(axis=0), index=feats).sort_values(ascending=False)
    return {"unit": unit, "rows": len(Xi), "top": {k: f"{float(v):.3f}" for k, v in mean_abs.head(10).items()}}


# ----------------------------------------------------------------------------- report
def main(argv: list[str] | None = None) -> int:
    setup_console()
    ap = argparse.ArgumentParser(description="StockWise AI: residual analysis.")
    ap.add_argument("--model", help="model name (default: the best official model)")
    ap.add_argument("--shap-rows", type=int, default=3000, help="validation rows used for SHAP")
    ap.add_argument("--no-shap", action="store_true")
    args = ap.parse_args(argv)

    d, model = load_predictions(args.model)
    d = add_flags(d)
    total_err, total_act = float(d["err"].sum()), float(d["actual_sales"].sum())
    print(f"Residual analysis of {model}: {len(d):,} validation rows, {d['date'].min().date()} -> {d['date'].max().date()}")
    print("Charts:")
    avp = plot_actual_vs_predicted(d, model)
    dist = plot_residual_distribution(d, model)
    day = plot_residuals_over_time(d, model)
    store = plot_error_by(d, "store_nbr", f"Error by store - {model}", "res_04_error_by_store.png")
    fam = plot_error_by(d, "family", f"Error by product family - {model}", "res_05_error_by_family.png")
    hol = plot_holidays(d, model)
    imp = plot_importance(model)
    shap_info = None if args.no_shap else run_shap(model, d, args.shap_rows)

    L: list[str] = []
    a = L.append
    a(f"# Residual analysis - {model} (auto-generated by src/residual_analysis.py)")
    a("")
    a(f"Validation {d['date'].min().date()} -> {d['date'].max().date()}, **{len(d):,} rows**. Residual = actual - predicted "
      "(positive = the model predicted too little). The test slice is not used.")
    a(f"Overall: **WAPE {wape(total_err, total_act):.2f}%**, MAE {d['err'].mean():.2f}, RMSE {np.sqrt((d['resid'] ** 2).mean()):.2f}.")
    a("")
    a("## 1. Actual vs predicted")
    a(f"- Total predicted / total actual = **{d['pred'].sum() / total_act:.3f}** "
      + ("(too low)" if d['pred'].sum() / total_act < 0.98 else "(too high)" if d['pred'].sum() / total_act > 1.02 else "(balanced)") + ".")
    a(f"- In the group with the SMALLEST sales the average prediction is {avp['small_ratio']:.2f} x the actual; "
      f"in the group with the LARGEST sales it is {avp['big_ratio']:.2f} x the actual "
      "(a value below 1 means the model under-predicts there).")
    a("")
    a("## 2. Residual distribution")
    a(f"- Mean residual {dist['mean']:+.2f}, median {dist['median']:+.2f}, standard deviation {dist['std']:.2f}, skewness {dist['skew']:.1f}.")
    a(f"- Predicted too little on {dist['under_%']:.0f}% of the rows, too much on {dist['over_%']:.0f}%, exactly right on {dist['exact_%']:.0f}%.")
    a("- A long tail on one side means a few large errors dominate the squared error (RMSE).")
    a("")
    a("## 3. Residuals over time")
    w = day["wape"].dropna()
    a(f"- Best day WAPE {w.min():.1f}% ({w.idxmin().date()}), worst day {w.max():.1f}% ({w.idxmax().date()}).")
    a(f"- Average daily bias {day['bias_%'].mean():+.1f}%. "
      + ("A bias that keeps the same sign on most days means the model is systematically too low or too high."
         if (np.sign(day['bias_%']).abs().mean() > 0 and abs(np.sign(day['bias_%']).mean()) > 0.6) else
         "The bias changes sign from day to day: no systematic drift."))
    a("")
    a("## 4. Error by store")
    st = store.sort_values("share_%", ascending=False)
    a(f"- The 5 stores with the largest share of the error: "
      + ", ".join(f"store {i} ({r['share_%']:.1f}%, WAPE {r['wape_%']:.0f}%)" for i, r in st.head(5).iterrows()) + ".")
    big = store[store["act"] >= store["act"].median()]
    if len(big):
        a(f"- Among the bigger stores the worst WAPE is store {big['wape_%'].idxmax()} ({big['wape_%'].max():.1f}%), the best store {big['wape_%'].idxmin()} ({big['wape_%'].min():.1f}%).")
    a("")
    a("## 5. Error by product family")
    ft = fam.sort_values("share_%", ascending=False)
    a("- The 5 families with the largest share of the error: "
      + ", ".join(f"{i} ({r['share_%']:.1f}%, WAPE {r['wape_%']:.0f}%)" for i, r in ft.head(5).iterrows()) + ".")
    vol = fam[fam["act"] >= fam["act"].median()]
    if len(vol):
        a(f"- Among the bigger families the hardest is {vol['wape_%'].idxmax()} (WAPE {vol['wape_%'].max():.1f}%), the easiest {vol['wape_%'].idxmin()} ({vol['wape_%'].min():.1f}%).")
    a("")
    a("## 6. Holidays vs normal days")
    a("")
    a("| group | rows | days | WAPE % | bias % | share of the error % |")
    a("|---|---|---|---|---|---|")
    for _, r in hol.iterrows():
        a(f"| {r['group'].strip()} | {int(r['rows']):,} | {int(r['days'])} | {r['wape_%']:.1f} | {r['bias_%']:+.1f} | {r['share_of_error_%']:.1f} |")
    few = hol[hol["group"].str.contains("holidays \\(any")]
    a("")
    if few.empty:
        a("The validation month has NO holiday day, so the behaviour on holidays cannot be judged from this analysis.")
    elif few["days"].iloc[0] <= 3:
        a(f"Careful: the validation month only has {int(few['days'].iloc[0])} holiday day(s), so the holiday numbers are weak evidence.")
    a("")
    a("## 7. Feature importance and SHAP")
    if imp:
        a("- Built-in importance, top 10: " + ", ".join(f"`{k}` ({v:.3f})" for k, v in imp.items()) + ".")
    if shap_info:
        a(f"- SHAP on {shap_info['rows']:,} random validation rows. Values are in **{shap_info['unit']}**. "
          "Top clues by average absolute SHAP value: " + ", ".join(f"`{k}` ({v})" for k, v in shap_info["top"].items()) + ".")
        a("- In the summary chart each dot is one row: red = a high value of the clue, blue = a low one; "
          "dots to the right push the prediction UP, to the left push it DOWN.")
    elif not args.no_shap:
        a("- SHAP was skipped (see the message in the terminal: usually `pip install shap`, or the model saved by train_models.py is a different one).")
    a("")
    a("Charts: " + ", ".join(f"`reports/figures/{p.name}`" for p in sorted(FIG_DIR.glob("res_*.png"))))
    out = config.REPORTS_DIR / "residual_analysis.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
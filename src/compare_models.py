
"""STEP 9: Model comparison and reporting.

Run from the project root:
    python src/compare_models.py

Reads saved training artifacts. Does NOT retrain models.

Inputs:
    reports/model_comparison_results.parquet
    reports/model_comparison_metadata.json

Outputs:
    reports/model_comparison.md
    reports/figures/01_rmsle_comparison.png
    reports/figures/02_speed_accuracy_tradeoff.png
    reports/figures/03_feature_importance_best.png
    reports/figures/04_feature_importance_all.png
    reports/figures/05_actual_vs_predicted.png
    reports/figures/06_residuals_distribution.png
    reports/figures/07_rmsle_by_family.png
    reports/figures/08_daily_forecast.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import config
from data_io import setup_console


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_COLOURS = {
    "LinearRegression": "#5b8db8",
    "RandomForest": "#e08e45",
    "XGBoost": "#c0392b",
}

DEFAULT_COLOUR = "#7f8c8d"


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def rmsle(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Root Mean Squared Log Error."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    predicted = np.clip(predicted, 0, None)
    actual = np.clip(actual, 0, None)

    return float(
        np.sqrt(
            np.mean(
                (np.log1p(predicted) - np.log1p(actual)) ** 2
            )
        )
    )


# ---------------------------------------------------------------------------
# Load saved artifacts
# ---------------------------------------------------------------------------

def load_artifacts() -> tuple[pd.DataFrame, dict]:
    """Load comparison predictions and model metadata."""

    results_path = config.MODEL_COMPARISON_RESULTS_PATH
    metadata_path = config.MODEL_COMPARISON_METADATA_PATH

    if not results_path.exists():
        raise FileNotFoundError(
            f"Comparison results not found: {results_path}\n"
            "Run train_models.py first."
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Comparison metadata not found: {metadata_path}\n"
            "Run train_models.py first."
        )

    print("[COMPARE] Loading comparison results...")
    comparison_df = pd.read_parquet(results_path)

    print("[COMPARE] Loading model metadata...")
    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    required_columns = {
        "id",
        "date",
        "store_nbr",
        "family",
        "actual_sales",
        "model",
        "predicted_sales",
    }

    missing = required_columns - set(comparison_df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in comparison results: {sorted(missing)}"
        )

    if comparison_df.empty:
        raise ValueError("Comparison results file is empty.")

    if not metadata.get("models"):
        raise ValueError("No model metadata found.")

    if not metadata.get("best_model"):
        raise ValueError("Best model is missing from metadata.")

    comparison_df["date"] = pd.to_datetime(
        comparison_df["date"]
    )

    print(
        f"[COMPARE] Rows: {len(comparison_df):,} "
        f"| Models: {comparison_df['model'].nunique()}"
    )

    return comparison_df, metadata


# ---------------------------------------------------------------------------
# Prepare model data
# ---------------------------------------------------------------------------

def prepare_model_data(
    comparison_df: pd.DataFrame,
    metadata: dict,
) -> tuple[dict, dict, str]:
    """Prepare predictions and metrics for each model."""

    metrics = {
        model["name"]: model
        for model in metadata["models"]
    }

    model_data = {}

    for model_name, group in comparison_df.groupby("model"):
        group = group.sort_values("id").reset_index(drop=True)

        actual = group["actual_sales"].to_numpy(dtype=float)
        predicted = np.clip(
            group["predicted_sales"].to_numpy(dtype=float),
            0,
            None,
        )

        model_data[model_name] = {
            "data": group,
            "actual": actual,
            "predicted": predicted,
        }

    best_name = metadata["best_model"]

    if best_name not in model_data:
        raise ValueError(
            f"Best model '{best_name}' has no saved predictions."
        )

    if best_name not in metrics:
        raise ValueError(
            f"Best model '{best_name}' has no metadata."
        )

    return model_data, metrics, best_name


# ---------------------------------------------------------------------------
# Chart utilities
# ---------------------------------------------------------------------------

def save_chart(fig, filename: str) -> None:
    """Save a chart inside reports/figures/."""

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()

    output_path = config.FIGURES_DIR / filename

    fig.savefig(
        output_path,
        dpi=130,
        bbox_inches="tight",
    )

    import matplotlib.pyplot as plt
    plt.close(fig)

    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Chart 1: RMSLE comparison
# ---------------------------------------------------------------------------

def plot_rmsle_comparison(metrics: dict) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    names = list(metrics.keys())
    scores = [metrics[name]["rmsle"] for name in names]

    colours = [
        MODEL_COLOURS.get(name, DEFAULT_COLOUR)
        for name in names
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    bars = ax.bar(
        names,
        scores,
        color=colours,
        width=0.5,
        zorder=3,
    )

    ax.set_title("Validation RMSLE — lower is better")
    ax.set_ylabel("RMSLE")
    ax.yaxis.set_major_formatter(
        mticker.FormatStrFormatter("%.4f")
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
        zorder=0,
    )

    max_score = max(scores)

    ax.set_ylim(0, max_score * 1.25)

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            score + max_score * 0.01,
            f"{score:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    best_index = int(np.argmin(scores))

    bars[best_index].set_edgecolor("gold")
    bars[best_index].set_linewidth(3)

    ax.annotate(
        "Best",
        xy=(best_index, scores[best_index]),
        xytext=(best_index, scores[best_index] + max_score * 0.09),
        ha="center",
        fontsize=11,
        color="goldenrod",
        fontweight="bold",
    )

    save_chart(fig, "01_rmsle_comparison.png")


# ---------------------------------------------------------------------------
# Chart 2: Speed vs accuracy
# ---------------------------------------------------------------------------

def plot_speed_accuracy(metrics: dict) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))

    for name, result in metrics.items():
        colour = MODEL_COLOURS.get(name, DEFAULT_COLOUR)

        ax.scatter(
            result["time_s"],
            result["rmsle"],
            s=200,
            color=colour,
            zorder=5,
            label=name,
        )

        ax.annotate(
            name,
            (result["time_s"], result["rmsle"]),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=9,
        )

    ax.set_title("Speed vs Accuracy trade-off")
    ax.set_xlabel("Training time (seconds)")
    ax.set_ylabel("Validation RMSLE (lower is better)")

    ax.invert_yaxis()
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend()

    save_chart(fig, "02_speed_accuracy_tradeoff.png")


# ---------------------------------------------------------------------------
# Chart 3: Best model feature importance
# ---------------------------------------------------------------------------

def plot_best_feature_importance(
    metrics: dict,
    best_name: str,
) -> None:
    import matplotlib.pyplot as plt

    importances = metrics[best_name].get("importances", {})

    if not importances:
        print("  Chart 03 skipped: no feature importances.")
        return

    top20 = sorted(
        importances.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:20]

    feature_names = [item[0] for item in top20]
    values = [item[1] for item in top20]

    fig, ax = plt.subplots(
        figsize=(9, max(5, 0.32 * len(feature_names)))
    )

    ax.barh(
        feature_names[::-1],
        values[::-1],
        color=MODEL_COLOURS.get(best_name, DEFAULT_COLOUR),
    )

    ax.set_title(
        f"Top 20 Feature Importances — {best_name}"
    )
    ax.set_xlabel("Importance")

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.4,
    )

    save_chart(fig, "03_feature_importance_best.png")


# ---------------------------------------------------------------------------
# Chart 4: Feature importance comparison
# ---------------------------------------------------------------------------

def plot_all_feature_importances(metrics: dict) -> None:
    import matplotlib.pyplot as plt

    models_with_importances = {
        name: result.get("importances", {})
        for name, result in metrics.items()
        if result.get("importances")
    }

    if not models_with_importances:
        print("  Chart 04 skipped: no feature importances.")
        return

    # Collect the union of each model's top 10 features.
    all_top_features = set()

    for importances in models_with_importances.values():
        top10 = sorted(
            importances.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:10]

        all_top_features.update(
            feature for feature, _ in top10
        )

    all_top_features = sorted(all_top_features)

    x = np.arange(len(all_top_features))

    model_count = len(models_with_importances)
    width = 0.8 / model_count

    fig, ax = plt.subplots(
        figsize=(max(10, len(all_top_features) * 0.6), 5)
    )

    for index, (name, importances) in enumerate(
        models_with_importances.items()
    ):
        # Normalize each model's importances separately.
        # This makes the chart easier to compare visually,
        # but does not make the underlying methods identical.
        total = sum(abs(value) for value in importances.values())

        normalized = {
            feature: abs(value) / total
            for feature, value in importances.items()
        } if total else {}

        values = [
            normalized.get(feature, 0.0)
            for feature in all_top_features
        ]

        offset = (
            index - model_count / 2 + 0.5
        ) * width

        ax.bar(
            x + offset,
            values,
            width=width * 0.9,
            label=name,
            color=MODEL_COLOURS.get(name, DEFAULT_COLOUR),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        all_top_features,
        rotation=45,
        ha="right",
        fontsize=8,
    )

    ax.set_title(
        "Normalized Feature Importance Comparison — All Models"
    )
    ax.set_ylabel("Normalized importance")

    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    save_chart(fig, "04_feature_importance_all.png")


# ---------------------------------------------------------------------------
# Chart 5: Actual vs predicted
# ---------------------------------------------------------------------------

def plot_actual_vs_predicted(
    model_data: dict,
    best_name: str,
) -> None:
    import matplotlib.pyplot as plt

    best = model_data[best_name]

    actual = best["actual"]
    predicted = best["predicted"]

    cap = float(np.percentile(actual, 99))
    mask = actual <= cap

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(
        actual[mask],
        predicted[mask],
        alpha=0.08,
        s=4,
        color=MODEL_COLOURS.get(best_name, DEFAULT_COLOUR),
    )

    limit = max(
        actual[mask].max(),
        predicted[mask].max(),
    ) * 1.05

    ax.plot(
        [0, limit],
        [0, limit],
        "k--",
        linewidth=1,
        label="Perfect prediction",
    )

    ax.set_title(
        f"Actual vs Predicted — {best_name}\n"
        f"(Actual values capped at 99th percentile = {cap:,.0f})"
    )

    ax.set_xlabel("Actual sales")
    ax.set_ylabel("Predicted sales")

    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.3)

    save_chart(fig, "05_actual_vs_predicted.png")


# ---------------------------------------------------------------------------
# Chart 6: Residual distribution
# ---------------------------------------------------------------------------

def plot_residual_distribution(
    model_data: dict,
    metrics: dict,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))

    for name, result in model_data.items():
        actual = result["actual"]
        predicted = result["predicted"]

        log_error = (
            np.log1p(predicted) - np.log1p(actual)
        )

        ax.hist(
            log_error,
            bins=120,
            alpha=0.5,
            color=MODEL_COLOURS.get(name, DEFAULT_COLOUR),
            label=(
                f"{name} "
                f"(RMSLE={metrics[name]['rmsle']:.4f})"
            ),
            density=True,
        )

    ax.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1,
    )

    ax.set_title(
        "Log-error Distribution "
        "(log1p predicted − log1p actual)"
    )

    ax.set_xlabel("Log error")
    ax.set_ylabel("Density")

    ax.legend()
    ax.grid(linestyle="--", alpha=0.3)

    save_chart(fig, "06_residuals_distribution.png")


# ---------------------------------------------------------------------------
# Chart 7: RMSLE by product family
# ---------------------------------------------------------------------------

def plot_rmsle_by_family(
    model_data: dict,
    metrics: dict,
    best_name: str,
) -> None:
    import matplotlib.pyplot as plt

    best_df = model_data[best_name]["data"].copy()

    best_df["predicted_sales"] = np.clip(
        best_df["predicted_sales"].to_numpy(dtype=float),
        0,
        None,
    )

    family_scores = {}

    for family, group in best_df.groupby("family"):
        family_scores[family] = rmsle(
            group["actual_sales"].to_numpy(),
            group["predicted_sales"].to_numpy(),
        )

    family_scores = dict(
        sorted(
            family_scores.items(),
            key=lambda item: item[1],
        )
    )

    fig, ax = plt.subplots(
        figsize=(9, max(5, 0.3 * len(family_scores)))
    )

    ax.barh(
        list(family_scores.keys()),
        list(family_scores.values()),
        color=MODEL_COLOURS.get(best_name, DEFAULT_COLOUR),
    )

    overall_rmsle = metrics[best_name]["rmsle"]

    ax.axvline(
        overall_rmsle,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"Overall RMSLE = {overall_rmsle:.4f}",
    )

    ax.set_title(
        f"RMSLE by Product Family — {best_name}"
    )
    ax.set_xlabel("RMSLE")

    ax.legend()
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    save_chart(fig, "07_rmsle_by_family.png")


# ---------------------------------------------------------------------------
# Chart 8: Daily actual vs predicted
# ---------------------------------------------------------------------------

def plot_daily_forecast(
    model_data: dict,
    best_name: str,
) -> None:
    import matplotlib.pyplot as plt

    best_df = model_data[best_name]["data"].copy()

    daily = (
        best_df.groupby("date")[
            ["actual_sales", "predicted_sales"]
        ]
        .sum()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(12, 4.5))

    ax.plot(
        daily.index,
        daily["actual_sales"],
        linewidth=1.6,
        color="#2c3e50",
        label="Actual total sales",
    )

    ax.plot(
        daily.index,
        daily["predicted_sales"],
        linewidth=1.6,
        linestyle="--",
        color=MODEL_COLOURS.get(best_name, DEFAULT_COLOUR),
        label=f"{best_name} forecast",
    )

    ax.fill_between(
        daily.index,
        daily["actual_sales"].to_numpy(),
        daily["predicted_sales"].to_numpy(),
        alpha=0.12,
        color=MODEL_COLOURS.get(best_name, DEFAULT_COLOUR),
    )

    ax.set_title(
        f"Daily Total Sales — Actual vs {best_name}"
    )

    ax.set_ylabel("Total sales")
    ax.legend()
    ax.grid(linestyle="--", alpha=0.3)

    save_chart(fig, "08_daily_forecast.png")


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def generate_markdown_report(
    metadata: dict,
    metrics: dict,
    model_data: dict,
    best_name: str,
) -> None:
    """Generate the model comparison Markdown report."""

    report_lines = [
        "# StockWise AI — Model Comparison Report",
        "",
        "## 1. Experiment Overview",
        "",
        f"- Training end: {metadata['train_end']}",
        f"- Validation start: {metadata['val_start']}",
        f"- Validation end: {metadata['val_end']}",
        f"- Number of features: {len(metadata['features'])}",
        f"- Validation rows: {len(model_data[best_name]['actual']):,}",
        f"- Best model: **{best_name}**",
        "",
        "## 2. Model Performance",
        "",
        "| Model | RMSLE | Training Time (seconds) |",
        "|---|---:|---:|",
    ]

    sorted_models = sorted(
        metrics.items(),
        key=lambda item: item[1]["rmsle"],
    )

    for name, result in sorted_models:
        marker = " **(Best)**" if name == best_name else ""

        report_lines.append(
            f"| {name}{marker} "
            f"| {result['rmsle']:.5f} "
            f"| {result['time_s']:.1f} |"
        )

    report_lines.extend([
        "",
        "## 3. Best Model",
        "",
        f"The best model is **{best_name}**, "
        f"with a validation RMSLE of "
        f"**{metrics[best_name]['rmsle']:.5f}**.",
        "",
        "## 4. Feature Importance",
        "",
        "Feature importance values are model-specific. "
        "The comparison chart normalizes each model's values "
        "separately for visual comparison; this does not make "
        "the underlying importance methods directly equivalent.",
        "",
    ])

    best_importances = metrics[best_name].get("importances", {})

    if best_importances:
        report_lines.extend([
            "| Rank | Feature | Importance |",
            "|---:|---|---:|",
        ])

        top_features = sorted(
            best_importances.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:20]

        for rank, (feature, value) in enumerate(
            top_features,
            start=1,
        ):
            report_lines.append(
                f"| {rank} | {feature} | {value:.6f} |"
            )
    else:
        report_lines.append(
            "Feature importance is not available for the best model."
        )

    report_lines.extend([
        "",
        "## 5. Generated Charts",
        "",
        "| Chart | Description |",
        "|---|---|",
        "| 01 | RMSLE comparison |",
        "| 02 | Training speed vs accuracy |",
        "| 03 | Best model feature importance |",
        "| 04 | Normalized feature importance comparison |",
        "| 05 | Actual vs predicted sales |",
        "| 06 | Log-error distribution |",
        "| 07 | RMSLE by product family |",
        "| 08 | Daily actual vs predicted sales |",
        "",
        "Charts are saved in `reports/figures/`.",
        "",
    ])

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    config.MODEL_COMPARISON_PATH.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print(f"[COMPARE] Report saved: {config.MODEL_COMPARISON_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    setup_console()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot  # noqa: F401
    except ImportError:
        print(
            "[ERROR] matplotlib is missing. "
            "Install it using: pip install matplotlib"
        )
        return 1

    try:
        comparison_df, metadata = load_artifacts()

        model_data, metrics, best_name = prepare_model_data(
            comparison_df,
            metadata,
        )

        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        print(f"\n[COMPARE] Best model: {best_name}")
        print("\n[COMPARE] Generating charts...")

        plot_rmsle_comparison(metrics)
        plot_speed_accuracy(metrics)
        plot_best_feature_importance(metrics, best_name)
        plot_all_feature_importances(metrics)
        plot_actual_vs_predicted(model_data, best_name)
        plot_residual_distribution(model_data, metrics)
        plot_rmsle_by_family(model_data, metrics, best_name)
        plot_daily_forecast(model_data, best_name)

        print("\n[COMPARE] Generating Markdown report...")

        generate_markdown_report(
            metadata,
            metrics,
            model_data,
            best_name,
        )

        print("\n" + "=" * 60)
        print("[COMPARE] Comparison completed successfully!")
        print("=" * 60)

        print(f"  Report  : {config.MODEL_COMPARISON_PATH}")
        print(f"  Figures : {config.FIGURES_DIR}")

        return 0

    except Exception as error:
        print(f"\n[ERROR] Comparison failed: {error}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())

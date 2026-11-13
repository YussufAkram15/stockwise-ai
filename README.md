# StockWise AI

Demand forecasting for a supermarket chain: we predict how many units each store will sell of each product family, **16 days ahead**.

DEPI Machine Learning capstone, team of five. Data: the Kaggle "Store Sales – Time Series Forecasting" competition (Corporación Favorita, Ecuador) — 54 stores, 33 product families, daily sales from 2013-01-01 to 2017-08-15.

## Team

| Member | Part | Model |
|---|---|---|
| Yussuf (team lead) | Naive baseline | Naive lag 21 |
| Hazem | Exploratory data analysis (EDA) | Linear Regression |
| Zenab | Data cleaning | Polynomial Regression |
| Adham | Feature engineering | Random Forest |
| Sara | Split, metrics, MLflow | XGBoost |

## Rules everyone follows

- **Horizon:** 16 days.
- **Split by date, never shuffled:** train up to 2017-06-30 · validation 2017-07-01 to 2017-07-30 · test 2017-07-31 to 2017-08-15.
- **Metrics:** MAE, RMSE, WAPE. WAPE is the main one (lower is better).
- **Lags:** 16, 21 and 28 days. **Rolling means:** 7 and 28 days, shifted by 16 days. Both are computed per store and product family.
- **No data leakage:** nothing newer than 16 days before the forecast day may be used. `transactions.csv` is never used as a feature.
- **MLflow:** every model run is logged.
- **Test set:** used once, at the end, for the chosen model.

## Setup

You need Python 3.12 or newer and Git.

```bash
git clone https://github.com/YussufAkram15/stockwise-ai.git
cd stockwise-ai
python -m venv .venv
```

Activate the environment:

```bash
source .venv/bin/activate        # Linux / macOS
```

```powershell
.venv\Scripts\activate           # Windows
```

Install the libraries:

```bash
pip install -r requirements.txt
```

## Get the data

The raw data is **not** in this repository (it is too big, and Kaggle's rules say each user downloads it).

1. Open <https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data> and accept the competition rules.
2. Download the files and put them in `data/raw/`:

```
data/raw/train.csv
data/raw/test.csv
data/raw/stores.csv
data/raw/oil.csv
data/raw/holidays_events.csv
data/raw/transactions.csv
data/raw/sample_submission.csv
```

## Run

Run every command from the project folder.

**1. Look at the raw files** (changes nothing):

```bash
python src/inspect_data.py
```

**2. Clean the data and check the result:**

```bash
python src/prepare_data.py
```

This writes the cleaned tables to `data/processed/` (main file: `train_clean.parquet`) and then runs `src/validate_data.py`, which checks that no row was lost or changed. The text reports go to `data/reports/` (copies of the last run are kept in `reports/`).

> `data/processed/` holds large generated files. Anyone can re-create them with this command, so please do not commit them.

**3. Draw the EDA charts:**

```bash
python src/eda.py
```

**4. Naive baseline:** open `notebooks/02_naive_baseline_yussuf.ipynb` and run all cells. It needs only `data/raw/train.csv`.

**5. Open the experiment logbook:**

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open the link it prints. `mlflow.db` is created on your own computer the first time a run is logged; it is not shared through Git.

## The number to beat

The naive baseline predicts "the same sales as 21 days ago" (same weekday, three weeks earlier). A model is only useful if its validation WAPE is lower than this.

| | MAE | RMSE | WAPE |
|---|---|---|---|
| Validation | 85.54 | 318.98 | 17.50% |
| Test | 92.72 | 335.98 | 19.85% |

These numbers are produced by the baseline notebook and saved in `reports/naive_baseline_metrics.csv`.

## Status (13 November 2026)

| Step | Owner | In this repository |
|---|---|---|
| Data inspection, cleaning, validation | Zenab | Done — `src/inspect_data.py`, `src/prepare_data.py`, `src/validate_data.py` |
| EDA charts and notes | Hazem | A version came in with the cleaning work: `src/eda.py`, `reports/eda/`. Hazem's own version is not in the repository yet |
| Naive baseline | Yussuf | Done — `notebooks/02_naive_baseline_yussuf.ipynb` |
| Feature engineering | Adham | Not yet |
| Split, metrics and MLflow helpers | Sara | Not yet |
| Linear, Polynomial, Random Forest, XGBoost | Hazem, Zenab, Adham, Sara | Not yet |
| Model comparison, residual analysis, 16-day forecast file | Sara, Yussuf | Not yet |

## Folders

```
data/raw/         raw Kaggle files (not in Git)
data/processed/   cleaned files made by src/prepare_data.py (generated, not committed)
docs/             project documents (requirements, planning, literature review)
notebooks/        one notebook per piece of work
reports/          generated reports and figures
src/              reusable Python code
```
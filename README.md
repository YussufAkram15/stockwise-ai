# StockWise AI — Retail Demand Forecasting

StockWise AI predicts **how many units each supermarket store will sell of each product family, for each of the next 16 days**. It uses four and a half years of real daily sales from a supermarket chain in Ecuador, together with promotions, holidays, oil prices and store information.

This is the DEPI AWS Machine Learning capstone of a five-person team. This README is written so that **someone who has never seen the project** can understand what it does, why it is built this way, and run it on their own computer.

---

## Table of contents

1. [The problem in plain words](#1-the-problem-in-plain-words)
2. [The data](#2-the-data)
3. [Words you need to know](#3-words-you-need-to-know)
4. [How the project works, step by step](#4-how-the-project-works-step-by-step)
5. [The rules the whole team follows](#5-the-rules-the-whole-team-follows)
6. [Results so far](#6-results-so-far)
7. [Project structure](#7-project-structure)
8. [Installation](#8-installation)
9. [How to run the project](#9-how-to-run-the-project)
10. [What each step produces](#10-what-each-step-produces)
11. [Troubleshooting](#11-troubleshooting)
12. [Team and status](#12-team-and-status)
13. [Limitations and next steps](#13-limitations-and-next-steps)
14. [Technologies used](#14-technologies-used)
15. [Deploy on AWS (Level 4)](#15-deploy-on-aws-level-4)

---

## 1. The problem in plain words

A supermarket has to order stock **before** it sells it.

- Order **too little** → empty shelves, lost sales, unhappy customers.
- Order **too much** → food expires, money is stuck in the storeroom.

So the manager wants to know: *"How much will each store sell of each type of product over the next two weeks?"* Guessing the future from the past like this is called a **forecast**.

**What we predict:** the number of units sold (`sales`) for every **store × product family × day**.
**How far ahead:** **16 days** (the length of the Kaggle test period, and a realistic ordering window).
**Type of problem:** **regression** (we predict a number), on **time-series** data (the order of the days matters).

Example of one real row we try to predict:

| date | store_nbr | family | sales | onpromotion |
|---|---|---|---|---|
| 2017-08-15 | 44 | GROCERY I | 8123 | 37 |

*"On 15 August 2017, store 44 sold 8,123 units of GROCERY I items, while 37 items of that family were on promotion."*

---

## 2. The data

**Source:** the Kaggle competition [Store Sales – Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data) (supermarket chain Corporación Favorita, Ecuador).

| Property | Value |
|---|---|
| Training rows | 3,000,888 |
| Stores | 54 |
| Product families (e.g. GROCERY I, DAIRY, BOOKS) | 33 |
| Series (one store × one family) | 54 × 33 = **1,782** |
| Dates | 2013-01-01 → 2017-08-15 (1,684 days; 25 December is missing every year because stores are closed) |
| Kaggle test period | 2017-08-16 → 2017-08-31 (16 days, 28,512 rows, no sales given) |

### The files

| File | What is in it | Used? |
|---|---|---|
| `train.csv` | Daily sales: date, store, family, sales, onpromotion | ✅ The main file |
| `stores.csv` | City, state, store type (A–E), cluster of each store | ✅ Joined to every row |
| `oil.csv` | Daily oil price (Ecuador's economy depends on oil) | ✅ Joined, with gaps filled |
| `holidays_events.csv` | National, regional and local holidays and events | ✅ Turned into 0/1 flags |
| `test.csv` | The 16 future days Kaggle asks for (no sales) | Optional (features for a Kaggle submission) |
| `transactions.csv` | Number of customers per store per day | ❌ **Never used** — on a day we forecast, the number of customers is not known yet, so using it would be *leakage* (see below) |
| `sample_submission.csv` | Kaggle answer format | Not used |

The data is **not** stored in this repository (it is large, and Kaggle's rules say each user downloads it). Section 8 explains how to get it.

---

## 3. Words you need to know

| Word | Meaning in this project |
|---|---|
| **Target** | The number we predict: `sales`. |
| **Feature** | A clue the model uses to predict the target (day of week, promotion, sales some days ago …). |
| **Series** | The daily sales of one store × one family, e.g. store 44 DAIRY. |
| **Horizon** | How far ahead we forecast: 16 days. |
| **Lag** | The sales of the same series *N* days earlier, copied onto today's row. A model reads one row at a time and cannot "look back", so lags put the history on the row. `sales_lag_21` on 15 August = sales on 25 July. |
| **Rolling mean** | The average of several days of sales, e.g. 7 days. More stable than one single day. |
| **Leakage** | Using information the model would **not** have at forecast time. It makes scores look great and fails in real life. |
| **Train / validation / test** | Train = the past the model learns from. Validation = "practice exam" used to compare and tune models. Test = "final exam", used **once** at the end. |
| **Baseline** | A deliberately simple forecast. Every real model must beat it, or it is not worth using. |
| **MAE, RMSE, WAPE** | Error scores (lower is better), explained in section 4.7. |
| **MLflow** | A logbook that records every model run: its settings and its scores. |

### Why only lags of 16 days or more?

Picture the real situation: it is the evening of 30 July, we know every sale up to today, and we must forecast 31 July → 15 August. For the **last** day, 15 August:

| Lag | Copies sales from | Known on 30 July? |
|---|---|---|
| 7 days | 8 August | ❌ future |
| 14 days | 1 August | ❌ future |
| **16 days** | 30 July | ✅ |
| **21 days** | 25 July | ✅ and the **same weekday** |
| **28 days** | 18 July | ✅ and the same weekday |

One model with the same columns is used for all 16 days, so a clue must exist for the farthest day. That is why **every sales-based feature is at least 16 days old**, and why rolling means are **shifted by 16 days**.

---

## 4. How the project works, step by step

```text
 raw CSV files (data/raw/)
        │
        ▼
 1. Inspect the raw files ............ src/inspect_data.py
        │
        ▼
 2. Clean + merge ..................... src/prepare_data.py ──► data/processed/train_clean.parquet
        │
        ▼
 3. Validate the cleaned data ......... src/validate_data.py (runs automatically after step 2)
        │
        ▼
 4. Explore the data (EDA charts) ..... src/eda.py
        │
        ▼
 5. Naive baseline (number to beat) ... notebooks/02_naive_baseline_yussuf.ipynb
        │
        ▼
 6. Build features (clues) ............ src/feature_engineering.py ──► data/processed/train_features.parquet
        │
        ▼
 7. Prove there is no leakage ......... src/leakage_check.py
        │
        ▼
 8. Train the models .................. src/train_models.py (Linear, Random Forest, XGBoost)
                                        src/train_polynomial.py (Polynomial)
        │
        ▼
 9. Compare + error analysis .......... src/final_comparison.py, src/residual_analysis.py
        │
        ▼
10. Score the best model on the test period once  src/test_best_model.py
```

One command, `python src/run_pipeline.py`, runs cleaning + validation, features, the leakage check, all models and the comparison (steps 2, 3, 6, 7, 8 and the comparison in 9). EDA (4), the baseline notebook (5) and `residual_analysis.py` (9) are run on their own (see section 9).

All paths and team rules live in one file, `src/config.py`, so every script uses the same settings.

### 4.1 Inspect the raw files — `src/inspect_data.py`

Reads the seven CSV files **without changing anything** and reports their size, columns, missing values, duplicates and date ranges. Result: no blocking problems. Known quirks: oil prices have gaps, some holidays were moved to another day ("transferred"), and 25 December has no rows.
Report: `reports/inspection_report.txt`.

### 4.2 Clean and merge — `src/prepare_data.py`

**The golden rule: no row is added or removed, and `sales` is never changed. Cleaning only adds columns.** The input has 3,000,888 rows × 6 columns; the output has 3,000,888 rows × 25 columns. Lags need a complete calendar for every series, and every store × family needs a forecast, so nothing is deleted — unusual rows get a **flag** instead.

What it does:

1. **Stores:** joins city, state, store type and cluster on `store_nbr`. The store `type` column is renamed `store_type` so it does not clash with the holiday `type`.
2. **Oil:** oil prices exist only on market days. Missing days get the **last known price carried forward** (Saturday gets Friday's price). This uses only the past; averaging Friday and Monday would use Monday's price, which is not known yet on Saturday. The very first day (2013-01-01) takes the first real price. A flag `oil_is_filled` marks filled days.
3. **Holidays:** each holiday row becomes one category:
   - `Holiday`, `Transfer`, `Additional`, `Bridge` → real day off;
   - `Event` (Black Friday, Mother's Day, the 2016 earthquake …) → its own flag, not a day off;
   - `Work Day` (a normally free day worked to make up for a bridge day) → its own flag;
   - `transferred = True` → the holiday was **moved away** from that date, so that date is a **normal day**; the new date appears as type `Transfer`.

   National holidays apply to all stores, regional ones to stores in that state, local ones to stores in that city. Result: 7 calendar flags (`is_holiday`, `is_holiday_national`, `is_holiday_regional`, `is_holiday_local`, `is_event`, `is_earthquake`, `is_work_day`).
4. **Quality-control (QC) flags**: their limits and statistics are learned only from data up to **2017-06-30** (the end of the training period, so nothing from validation or test leaks in), then applied to every day:

   | Flag | Meaning | Share of rows |
   |---|---|---|
   | `qc_zero_series` | This store × family never sold anything (54 of 1,782 series) | 3.0% |
   | `qc_before_first_sale` | Day before the series' first ever sale (store not opened yet, family not stocked yet) | 18.5% |
   | `qc_store_closed_day` | The store sold nothing in any family that day | 8.1% |
   | `qc_sales_outlier` | Sale above Q3 + 3 × IQR of its own series (flagged, **not** removed — these are real sales) | 0.6% |
   | `qc_sales_suspect` | **Suspected data error**: above Q3 + 10 × IQR of its own series **and** not explained by a holiday, event, earthquake or promotion (1,954 rows, e.g. store 39 MEATS sold 89,576 on 2016-10-07, limit 311). Left out of **training** only | 0.07% |

   ⚠️ QC flags are built from the sales themselves, so they are **never used as model features** — only to filter rows.

   Extra column `sales_capped`: the same as `sales`, except suspected data errors are capped at the normal outlier limit. Lags and rolling means are built from it, so one bad day is not copied into the clues of later days. `sales` itself is never changed, and every score uses the real `sales`.
5. **Save:** sorted by store → family → date and written as Parquet (smaller and faster than CSV, keeps column types).

Report with every decision: `reports/cleaning_report.md`.

### 4.3 Validate — `src/validate_data.py`

A second, independent check. It re-reads the raw files and recomputes everything (row counts, ids, unchanged sales, complete calendar grid, oil filling, holiday flags, QC flags) instead of trusting the cleaning code. Each check prints `PASS`, `WARN` or `FAIL`. The latest run has **0 FAIL** and one expected `WARN`: 1 January 2017 is not a holiday in the official file (it was transferred), even though almost all stores were closed — `qc_store_closed_day` flags 53 of the 54 stores that day.
Report: `reports/validation_report.txt`.

### 4.4 Explore the data — `src/eda.py`

Draws nine charts (`reports/eda/01_…png` to `09_…png`) and writes notes with numbers computed from the data (`reports/eda/eda_notes.md`). Main findings:

- **Trend:** average daily sales doubled from 2013 to 2016 (+105%), partly because new stores opened.
- **Weekly pattern:** weekends sell **+39%** vs Monday–Friday (Sunday best, Thursday weakest) → the weekday is a strong clue.
- **Day of month:** paydays themselves add only about +1%, but the first days of each month are 7–17% above normal (people shop right after the end-of-month payday).
- **Concentration:** the top 3 families (GROCERY I, BEVERAGES, PRODUCE) make 64% of all units; the 10 biggest stores make 40%.
- **Promotions:** only recorded from 2014-04-01. Rows with a promotion sell 9.6× more on average, but most of that is because big families are promoted more often; inside the same store and family the typical boost is about 1.5×.
- **Holidays:** national holidays about +15%; 1 January about −99% (stores closed).
- **Earthquake (2016-04-16):** sales +16% in the two weeks after → a one-off, flagged with `is_earthquake`.
- **Zeros:** 31% of rows have zero sales (BOOKS is zero on 97% of days).

### 4.5 Naive baseline — `notebooks/02_naive_baseline_yussuf.ipynb`

The simplest possible forecast: **"sales will be the same as on the same weekday three weeks earlier"** (lag 21). In forecasting books this is called a *seasonal naive* forecast. Nothing is trained. It is the **number every model must beat**.

- Why 21 and not 7: with a 16-day horizon, sales 7 days before the last forecast day are not known yet (leakage).
- Why 21 and not 16: 16 lands on a different weekday. Copying from 16 days ago gives about **35%** WAPE on validation, copying from 21 days ago gives **17.5%**.
- Lags are built by **matching dates** (date + 21 days), not by counting rows, because the missing 25 December would shift a row count by one day.

### 4.6 Features — `src/feature_engineering.py`

Turns the cleaned table into model-ready clues. All statistics and encodings are fitted on the training period only.

| Group | Features |
|---|---|
| Calendar | `day_of_week`, `day_of_month`, `week_of_year`, `month`, `quarter`, `year`, `is_weekend`, `days_to_month_end`, `is_payday`, `days_to_next_holiday`, `days_since_last_holiday` (capped at 30) |
| Sales lags | `sales_lag_16`, `sales_lag_21`, `sales_lag_28` |
| Rolling means (shifted by 16 days) | `sales_roll_7_mean`, `sales_roll_28_mean` |
| Promotions (known in advance — stores plan them, and `test.csv` gives them) | `onpromotion`, `promo_lag_7`, `promo_roll_7_mean` |
| Oil (lagged so they are known 16 days ahead) | `oil_lag_16`, `oil_lag_21`, `oil_lag_28` |
| Holidays (the calendar is known in advance) | the 7 flags from step 4.2 |
| Store and family | `store_nbr`, `cluster`, `store_type_enc`, `family_enc`, `city_enc`, `state_enc`, `store_family_mean_sales` (series average, delayed by 16 days) |

Lags and rolling means are computed **per store and family** and by **matching dates**, so the missing 25 December does not shift them (when the exact day is missing, the day before is used — always older, never newer). Sales clues are built from `sales_capped` (see 4.2).
Report: `reports/feature_report.md`.

**Leakage check — `src/leakage_check.py`.** A proof, not a promise: it builds the features twice, the second time after scrambling every sale and oil price after the end of training. Every feature of every row up to 15 days after the end of training must come out **identical** — if one changes, that feature looked at the future and the check fails. As a control, later rows must change. It also checks that no model uses `sales`, ids, dates or `qc_` flags as a clue.
Report: `reports/leakage_report.txt`.

### 4.7 Models and evaluation — `src/train_models.py`, `src/train_polynomial.py`, `src/final_comparison.py`, `src/residual_analysis.py`

Five models are compared on exactly the same days:

| Model | Owner | In plain words |
|---|---|---|
| Naive lag 21 | Yussuf | Copy the sales from three weeks ago. |
| Linear Regression | Hazem | Draws the best straight-line relationship between the clues and sales. |
| Polynomial Regression | Zenab | Like Linear, but the line may curve (adds squared and multiplied clues). Used on a few numeric clues only, so the number of columns does not explode. |
| Random Forest | Adham | Many decision trees, each trained on a random part of the data, voting together. |
| XGBoost | Sara | Trees built one after another, each fixing the mistakes of the previous ones. |

**Error scores** (lower is better). Example with actual = [100, 10, 0] and predicted = [90, 20, 5]:

| Score | Formula | Example | Meaning |
|---|---|---|---|
| **MAE** | average of \|actual − predicted\| | (10 + 10 + 5) / 3 = 8.33 | "on average we are off by 8.33 units per row" |
| **RMSE** | √(average of (actual − predicted)²) | √75 = 8.66 | like MAE, but big misses count much more |
| **WAPE** (main score) | sum of \|actual − predicted\| ÷ sum of actual | 25 / 110 = 22.7% | "our total error is 22.7% of everything sold" |

Why WAPE is the main score: it is a percentage a manager understands; big families (most of the units) count most; and it works with zero-sales rows. The common MAPE divides each row by its own sales and breaks on the 31% of rows with zero sales.

How the scripts work together:

- `train_models.py` trains Linear Regression, Random Forest and XGBoost on the training period, scores them on validation, logs each run to MLflow and saves the best one to `models/best_model.pkl`. Linear Regression does not get the label codes (store number, family code, city code …), because a straight line would read "family 20" as twice "family 10".
- **XGBoost early stopping is honest:** the number of trees is chosen on the **last 28 days of the training period**, then the model is retrained on the whole training period. Validation is never used to tune it.
- `train_polynomial.py` trains the Polynomial model on a few numeric clues (degree 2).
- `final_comparison.py` scores **every** model, including the naive baseline, with the same code on the same validation rows, and writes `reports/model_comparison.md` / `.csv` and charts.
- `residual_analysis.py` explains where the best model is wrong: by store, by family, over time, on holidays, and which clues matter most.

None of these scripts touches the test period.

### 4.8 Level 2: from forecast to reorder advice — `src/forecast_best_model.py`, `src/inventory.py`

A forecast only helps if it turns into a decision: **"should this store reorder this product now?"**

1. `forecast_best_model.py` takes the best model (`models/best_model.pkl`, today Random Forest) and forecasts the next 16 days (2017-08-16 → 08-31) for every store × family, plus its validation predictions. (`forecast_naive.py` writes the same files with the naive baseline, so Level 2 also works without trained models.)
2. `inventory.py` applies standard stock formulas to every series:

```text
sigma          = how much the forecast usually misses per day (from the validation errors)
safety_stock   = 1.65 × sigma × √lead_time          (1.65 → enough 95% of the time)
reorder_point  = average daily forecast × lead_time + safety_stock
status         = REORDER    if stock on hand ≤ reorder_point
                 OVERSTOCK  if stock on hand > 16-day forecast + safety_stock
                 OK         otherwise
order_qty      = 16-day forecast + safety_stock − stock on hand   (when REORDER)
```

⚠️ **The Favorita data has no stock levels and no delivery times, so both are simulated** (lead time 2–7 days per store, stock 0–30 days of sales, fixed random seed). The table shows the **method**; a real shop would plug in its real stock.

A better model means a smaller sigma, so a smaller safety stock, so less money stuck on shelves — this is how Level 1 pays off in Level 2.
Output: `reports/reorder_table.csv` (one row per store × family) and `reports/inventory_summary.md`.

### 4.9 Level 3: dashboard — `app/dashboard.py`

A Streamlit web page that only **reads** the files above (no training, no raw data):

- **Overview** — how many series need REORDER / OK / OVERSTOCK, the 10 most urgent items, the total forecast per day.
- **Forecast** — pick a store and a family: the last 60 days of real sales, then the 16-day forecast with a shaded band (± the usual daily error, sigma), and its stock advice. Hover the (?) next to each number for a one-line explanation.
- **Reorder table** — filter by status, store, family and "runs out before", download as CSV. Answers questions like *"which stores must order BEVERAGES now?"* (status REORDER) or *"which stores may run out of BEVERAGES next week if nothing is ordered?"* (status REORDER + OK, runs out before 2017-08-23).
- **Model accuracy** — the final test score (Random Forest 15.87% vs baseline 19.85% on 16 unseen days), the comparison table of all models against the baseline, and real vs predicted in July for any series.

---

## 5. The rules the whole team follows

| Rule | Value |
|---|---|
| Forecast horizon | 16 days |
| Split by date, **never shuffled** | Train: up to 2017-06-30 · Validation: 2017-07-01 → 2017-07-30 · Test: 2017-07-31 → 2017-08-15 (set once in `src/config.py`: `TRAIN_END`, `VAL_START`, `VAL_END`, `TEST_START`, `TEST_END`) |
| Test period | Used **once**, at the end, for the chosen model only |
| Scores | MAE, RMSE, WAPE (WAPE is the main one) |
| Lags | 16, 21, 28 days, per store and family |
| Rolling means | 7 and 28 days, shifted by 16 days, per store and family |
| No leakage | Nothing newer than 16 days before the forecast day; `transactions.csv` never used; QC flags never used as features |
| MLflow | Every model run is logged: `sqlite:///mlflow.db`, experiment `stockwise-ai`, metrics named `val_mae`, `val_rmse`, `val_wape` (`test_…` for the final test score) |
| GitHub | One branch per person; changes reach `main` only through a reviewed Pull Request |

Why a random split is wrong here: shuffling would put future days (e.g. 10 July) into training while the model is scored on earlier days (5 July). The model would learn from the future, which can never happen in real life.

---

## 6. Results so far

**The number to beat — naive lag 21:**

| | MAE | RMSE | WAPE |
|---|---|---|---|
| Validation (2017-07-01 → 07-30) | 85.54 | 318.98 | **17.50%** |
| Test (2017-07-31 → 08-15) | 92.72 | 335.98 | **19.85%** |

*Read it as: "copying sales from three weeks ago, the total error is 17.5% of all units sold in July."* A model is only useful if its **validation WAPE is below 17.50%**. Saved in `reports/naive_baseline_metrics.csv`; chart in `reports/figures/naive_baseline_actual_vs_predicted.png`.

Where the baseline fails (and what the real models should fix): it knows the weekday but not the day of the month (it guesses 22–26% too low on 1–2 August, right after payday) and not holidays (it guesses 18% too high on the weekend after the 11 August national holiday).

**All models on validation** (2017-07-01 → 07-30, the same 53,460 rows for every model; from `reports/model_comparison.md`):

| Model | MAE | RMSE | WAPE | vs baseline |
|---|---|---|---|---|
| **Random Forest** (best) | **62.30** | **219.30** | **12.75%** | 27% less error |
| XGBoost | 67.47 | 260.20 | 13.81% | 21% less error |
| Polynomial Regression | 79.10 | 250.38 | 16.19% | 7.5% less error |
| Naive lag 21 (baseline) | 85.54 | 318.98 | 17.50% | — |
| Linear Regression | 86.77 | 269.53 | 17.76% | 1.4% **more** error |

What this says:

- **Random Forest wins**: its total error is 12.75% of all units sold, about a quarter less than copying the past. Its predictions are not biased (predicted total / actual total = 1.01).
- **Trees beat straight lines**: sales depend on combinations (this family, in this store, on a weekend, with a promotion), which trees capture and a single straight line cannot. Linear Regression does not even beat the baseline.
- **Where it is still wrong**: 84% of the error comes from the biggest 25% of series (BEVERAGES, GROCERY I, CLEANING); holidays are harder (WAPE 19.9% vs 12.7% on normal days); the largest single misses are sudden one-day jumps (e.g. store 30 CLEANING sold 10,392 on 2017-07-10 against a usual few hundred).

**Final test score — the honest number** (`reports/test_result.md`):

Random Forest was chosen on validation, retrained on everything up to 2017-07-30 (training + validation, as a shop standing on 30 July would do), then scored **once** on the 16 test days it had never seen (2017-07-31 → 08-15):

| | MAE | RMSE | WAPE |
|---|---|---|---|
| **Random Forest, test** | **74.14** | **255.33** | **15.87%** |
| Naive lag 21, test | 92.72 | 335.98 | 19.85% |

*Read it as: "on 16 days the model never saw, its total error is 15.9% of all units sold — 20% less error than copying the past."*

The test period is harder than validation for **both** the model and the baseline (the baseline also goes from 17.50% to 19.85%; only these two are scored on test): it contains the start-of-month payday days (1–2 August) and the national holiday of 10–11 August, which are harder to forecast than the normal July days used for validation.

---

## 7. Project structure

```text
stockwise-ai/
├── data/
│   ├── raw/                  ← you put the Kaggle CSV files here (not in Git)
│   └── processed/            ← created by the scripts (not in Git)
├── docs/                     ← project documents: version_1 (31 Oct) and version_2 (PDFs + infographic)
├── models/                   ← saved trained models, created by train_models.py (not in Git)
├── notebooks/
│   └── 02_naive_baseline_yussuf.ipynb
├── reports/
│   ├── inspection_report.txt
│   ├── cleaning_report.md
│   ├── validation_report.txt
│   ├── feature_report.md
│   ├── naive_baseline_metrics.csv
│   ├── leakage_report.txt
│   ├── model_comparison.md / .csv ← scores of every model on validation
│   ├── residual_analysis.md  ← where the best model is wrong
│   ├── test_result.md / .json← the one-time test score
│   ├── reorder_table.csv     ← Level 2: advice per store × family (+ inventory_summary.md)
│   ├── eda/                  ← 9 EDA charts + eda_notes.md
│   └── figures/              ← model charts
├── src/
│   ├── config.py             ← all paths, split dates and cleaning rules (one place)
│   ├── data_io.py            ← shared helpers (reading files, writing reports)
│   ├── inspect_data.py       ← step 1
│   ├── prepare_data.py       ← step 2 (also runs step 3)
│   ├── validate_data.py      ← step 3
│   ├── eda.py                ← step 4
│   ├── feature_engineering.py← step 6
│   ├── leakage_check.py      ← step 7 (proof of no leakage)
│   ├── train_models.py       ← step 8 (Linear, Random Forest, XGBoost)
│   ├── train_polynomial.py   ← step 8 (Polynomial)
│   ├── final_comparison.py   ← step 9 (one table for all models + charts)
│   ├── residual_analysis.py  ← step 9 (where the best model is wrong)
│   ├── run_pipeline.py       ← runs cleaning → comparison with one command
│   ├── test_best_model.py    ← step 10 (the best model on the test period, once)
│   ├── forecast_best_model.py← Level 2: 16-day forecast with the best model
│   ├── forecast_naive.py     ← Level 2: the same files with the naive baseline
│   └── inventory.py          ← Level 2: reorder advice per store × family
├── app/
│   └── dashboard.py          ← Level 3: the dashboard (streamlit run app/dashboard.py)
├── deploy/                   ← Level 4: Dockerfile, AWS boot script, copy of the dashboard's 6 result files
├── requirements.txt          ← exact library versions
└── README.md
```

---

## 8. Installation

### What you need

- **Python 3.12** (tested with 3.12.3). Check with `python --version` (on some systems `python3 --version`).
- **Git**.
- About **8 GB of RAM** and **2 GB of free disk space** (the data has 3 million rows).
- A free **Kaggle account** to download the data.

### Step 1 — Get the code

```bash
git clone https://github.com/YussufAkram15/stockwise-ai.git
cd stockwise-ai
```

### Step 2 — Create a virtual environment

A virtual environment is a private folder of libraries for this project, so it does not interfere with other Python projects on your computer.

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

> If PowerShell refuses to run the script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use Command Prompt with `.venv\Scripts\activate.bat`.

When it works, your terminal line starts with `(.venv)`. Activate it again every time you open a new terminal.

### Step 3 — Install the libraries

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The versions are pinned (exact numbers) so everyone gets the same results.

### Step 4 — Download the data

1. Open <https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data> and log in.
2. Click **"Late Submission"** or **"I Understand and Accept"** to accept the competition rules (otherwise the download is blocked).
3. Click **"Download All"** and unzip the file.
4. Put the CSV files in `data/raw/`:

```text
data/raw/train.csv
data/raw/test.csv
data/raw/stores.csv
data/raw/oil.csv
data/raw/holidays_events.csv
data/raw/transactions.csv
data/raw/sample_submission.csv
```

The cleaning needs at least `train.csv`, `stores.csv`, `oil.csv` and `holidays_events.csv`; `test.csv` is optional.

---

## 9. How to run the project

Run every command **from the project folder** (`stockwise-ai/`) with the virtual environment active.

**The short way — everything from the raw files to the comparison table in one command:**

```bash
python src/run_pipeline.py
```

It runs cleaning + validation, features, the leakage check, all models and the comparison, and stops at the first step that fails. Options: `--no-mlflow`, `--models lr rf` (only some of the three models in `train_models.py`), `--from train_models` (restart from a step), `--skip leakage_check`. Then run `python src/residual_analysis.py` for the error analysis.

**The long way — one step at a time**, in this order (each step needs the files made by the step before):

**Step 1 — Look at the raw files** (changes nothing):

```bash
python src/inspect_data.py
```

**Step 2 — Clean the data and validate it:**

```bash
python src/prepare_data.py
```

Writes `data/processed/train_clean.parquet` (and `test_clean.parquet`), then runs the validation automatically. It must end with **0 FAIL**. Options: `--csv` also writes a CSV copy; `--skip-validation` skips the checks. To run the checks alone: `python src/validate_data.py`.

**Step 3 — Draw the EDA charts:**

```bash
python src/eda.py
```

**Step 4 — Naive baseline:** open `notebooks/02_naive_baseline_yussuf.ipynb` and run all cells. It only needs `data/raw/train.csv`. Two ways to open it:

- **VS Code** with the *Python* and *Jupyter* extensions: open the file, pick the `.venv` interpreter as kernel, click **Run All**.
- **In the browser:** `pip install notebook`, then `jupyter notebook notebooks/02_naive_baseline_yussuf.ipynb`.

**Step 5 — Build the features:**

```bash
python src/feature_engineering.py
```

Writes `data/processed/train_features.parquet` and `reports/feature_report.md`. Add `--no-test` to skip the Kaggle test period.

**Step 6 — Check for leakage, train and compare the models** (see section 4.7):

```bash
python src/leakage_check.py
python src/train_models.py
python src/train_polynomial.py
python src/final_comparison.py
python src/residual_analysis.py
```

Options for `train_models.py`: `--models lr rf xgb` trains only some models (lr = Linear, rf = Random Forest, xgb = XGBoost); `--no-mlflow` skips the logbook. `residual_analysis.py --no-shap` skips the SHAP charts (they need `pip install shap`).

**Step 7 — The final test score** (only once, after the best model is chosen):

```bash
python src/test_best_model.py
```

It retrains the best model on training + validation and scores the test period once. It refuses to run a second time (the test period must be looked at only once); use `--force` only if the model or the data changed.

**Step 7b — Level 2, the reorder advice** (after the models are trained):

```bash
python src/forecast_best_model.py
python src/inventory.py
```

**Step 8 — Level 3, the dashboard:**

```bash
streamlit run app/dashboard.py
```

It opens in the browser (usually <http://localhost:8501>).

**Step 9 — Open the experiment logbook (MLflow):**

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open the link it prints (usually <http://127.0.0.1:5000>). `mlflow.db` is created on your computer the first time a run is logged; it is not shared through Git.

---

## 10. What each step produces

| Step | File | What it is |
|---|---|---|
| 1 | `reports/inspection_report.txt` | Facts about the raw files |
| 2 | `data/processed/train_clean.parquet` | Cleaned data: 3,000,888 rows × 25 columns (not in Git) |
| 2 | `data/processed/test_clean.parquet` | The Kaggle test period, enriched the same way (not in Git) |
| 2 | `reports/cleaning_report.md` | Every cleaning decision and a dictionary of all 25 columns |
| 3 | `reports/validation_report.txt` | All PASS / WARN / FAIL checks |
| 4 | `reports/eda/*.png`, `reports/eda/eda_notes.md` | 9 charts and the findings |
| 4 (baseline) | `reports/naive_baseline_metrics.csv`, `reports/figures/naive_baseline_actual_vs_predicted.png` | The number to beat |
| 5 | `data/processed/train_features.parquet`, `reports/feature_report.md` | The feature table and its summary |
| 6 | `reports/leakage_report.txt` | The leakage proof: PASS / FAIL per feature |
| 6 | `models/best_model.pkl` | The best trained model (re-created on each run, not in Git) |
| 6 | `reports/model_comparison.md`, `reports/model_comparison.csv`, `reports/figures/*.png` | One table for all models, plus charts |
| 6 | `reports/residual_analysis.md`, `reports/figures/res_*.png` | Where the best model is wrong |
| 7 | `reports/test_result.md`, `reports/test_result.json` | The final test score of the best model |
| Level 2 | `reports/forecast_16d.parquet`, `reports/recent_sales.parquet`, `reports/val_predictions.parquet` | 16-day forecast, last 60 real days, July predictions (`forecast_best_model.py`; parquet not in Git) |
| Level 2 | `reports/reorder_table.csv`, `reports/inventory_summary.md` | Reorder advice per store × family (`inventory.py`) |
| Level 3 | `app/dashboard.py` | The dashboard, reads the files above |
| Level 4 | `deploy/data/` | Copy of the 6 files the dashboard reads, for the Docker image (`deploy/refresh_data.sh`) |

Generated data and models are **not** committed: anyone can re-create them by running the steps above.

---

## 11. Troubleshooting

| Problem | Fix |
|---|---|
| `train_clean.parquet not found` | Run step 2 (`python src/prepare_data.py`) first. |
| `FileNotFoundError: data/raw/train.csv` | The CSV files are not in `data/raw/`, or you are not in the project folder. Run `cd stockwise-ai`. |
| `ModuleNotFoundError` (e.g. `pandas`, `pyarrow`) | The virtual environment is not active, or step 3 of the installation was skipped. Activate `.venv` and run `pip install -r requirements.txt`. |
| `pip` cannot find a version | Your Python is too old. Use Python 3.12. |
| The computer freezes or `MemoryError` | The data needs about 8 GB of RAM. Close other programs; run one step at a time. |
| Kaggle download says "rules not accepted" | Accept the competition rules on the Kaggle page first (installation step 4). |
| `mlflow ui` shows no runs | Start it from the project folder, where `mlflow.db` is, and run the baseline notebook at least once. |

---

## 12. Team and status

| Member | Part | Model |
|---|---|---|
| Yussuf (team lead) | Repository, naive baseline, integration, final pipeline | Naive lag 21 |
| Hazem | Exploratory data analysis (EDA) | Linear Regression |
| Zenab | Data cleaning, README | Polynomial Regression |
| Adham | Feature engineering | Random Forest |
| Sara | Split, scores, MLflow, model comparison, error analysis | XGBoost |

**Status (21 November 2026):**

| Step | Owner | Status |
|---|---|---|
| Inspection, cleaning, validation | Zenab | ✅ Done |
| EDA charts and notes | Hazem | ✅ A version is in the repository (`src/eda.py`); Hazem's own version is not yet |
| Naive baseline | Yussuf | ✅ Done, logged in MLflow as `naive_lag21` |
| Suspected data errors (`qc_sales_suspect`, `sales_capped`) | Zenab | ✅ Done |
| Feature engineering | Adham, Zenab | ✅ Done (lags 16/21/28, rolling 7/28 shifted by 16, holiday distance) |
| Leakage check | Zenab | ✅ Done — `src/leakage_check.py` |
| Split dates | Sara | ✅ Done — agreed train / validation / test dates in `src/config.py`, used by every script |
| Scores (MAE / RMSE / WAPE) and MLflow logging | Sara, Zenab | ✅ Done — every model logs `val_mae`, `val_rmse`, `val_wape` to the `stockwise-ai` experiment |
| Linear, Random Forest, XGBoost | Hazem, Adham, Sara | ✅ Done — `src/train_models.py` (XGBoost early stopping on training data only) |
| Polynomial Regression | Zenab | ✅ Done — `src/train_polynomial.py` |
| Model comparison and error analysis | Sara, Zenab | ✅ Done — `src/final_comparison.py`, `src/residual_analysis.py` |
| Final test score of the best model | Yussuf | ✅ Done — Random Forest, test WAPE 15.87% (baseline 19.85%), `src/test_best_model.py` |
| Level 2 — reorder advice | Yussuf | ✅ Done — `src/forecast_best_model.py`, `src/inventory.py` (stock simulated) |
| Level 3 — dashboard | Yussuf | ✅ Done — `app/dashboard.py` |
| Level 4 — deployment (Docker on AWS EC2) | Yussuf, Adham | ✅ Done — tested on EC2 (Frankfurt) on 2026-11-27, see section 15 |

---

## 13. Limitations and next steps

**Still to do:**

- The best model is trained on data up to 2017-06-30; for the real 16-day forecast it could be retrained on all data up to 2017-08-15.
- Level 4 for production: push the image to Amazon ECR and run it on ECS Fargate (HTTPS, no server to manage).

**Limitations:**

- Stock levels and lead times in Level 2 are simulated: the dataset has none.
- Sudden one-day jumps (e.g. a cleaning-products order 10× the usual) cannot be predicted from past sales; they cause the biggest single errors.
- Holidays are harder than normal days (WAPE 19.9% vs 12.7%).

---

## 14. Technologies used

| Technology | Used for |
|---|---|
| Python 3.12 | Everything |
| pandas, NumPy | Loading, merging and transforming the data |
| PyArrow | Reading and writing Parquet files |
| Matplotlib, seaborn | Charts |
| scikit-learn | Linear / Polynomial Regression, Random Forest, pipelines, scaling |
| XGBoost | Gradient-boosted trees |
| MLflow | Experiment logbook |
| Jupyter | The baseline notebook |
| Git and GitHub | Version control and team review |
| Streamlit, Altair | The dashboard and its charts |
| Docker, AWS EC2 | Deployment (Level 4) |

---

## 15. Deploy on AWS (Level 4)

**What goes online:** only the dashboard. The model runs offline on our laptops (batch). The dashboard reads 6 result files (1.4 MB), so the server needs no model, no raw data and no training libraries.

**How it works:**

1. `deploy/data/` holds a copy of the 6 files the dashboard reads (`reports/*.parquet` is not in Git, so the server could not get them otherwise).
2. `deploy/Dockerfile` builds a small image: Python 3.12 + Streamlit + the dashboard + those 6 files.
3. `deploy/ec2-user-data.sh` runs by itself when an EC2 server starts: it installs Docker and Git, clones this repository (`main`), builds the image and starts it on port 8501.
4. A **security group** opens only port 8501 to the internet.

```text
GitHub (main) ──git clone──▶ EC2 t3.micro ──docker build──▶ container :8501 ──▶ http://<public-ip>:8501
```

**Cost:** a `t3.micro` server + public IP + 8 GB disk is about $0.02 per hour, **under $1 per day of use**. We start it only for testing and for the demo, then terminate it ($0 when nothing runs).

### One-time setup (AWS console)

1. **Billing alert:** Billing → Budgets → create a monthly budget of **$5** with an e-mail alert. Work as an IAM user, not the root user.
2. **Region:** Europe (Frankfurt) `eu-central-1`.
3. **Security group** (EC2 → Security Groups → Create) named `stockwise-sg`: inbound rule *Custom TCP*, port **8501**, source `0.0.0.0/0`. No SSH rule is needed (see the troubleshooting step below if something goes wrong).
4. **Launch template** (EC2 → Launch Templates → Create) named `stockwise-dashboard`:
   - AMI: **Amazon Linux 2023**, instance type **t3.micro**, storage **8 GB gp3**
   - Network settings: Subnet = *Don't include in launch template*; Firewall → *Select existing security group* → `stockwise-sg`.
     Do **not** add a network interface under *Advanced network configuration*: the default network already gives the server a public IP, and setting the security group in two places makes the launch fail.
   - Advanced details → **User data**: paste the whole content of `deploy/ec2-user-data.sh`

### Start, check, stop

1. EC2 → Launch Templates → `stockwise-dashboard` → **Launch instance from template**.
2. Wait about **5 minutes** (the server installs Docker and builds the image).
3. Copy the instance's **Public IPv4 address** and open `http://<public-ip>:8501` (http, not https).
4. When finished: EC2 → Instances → select it → **Instance state → Terminate**. Check that no instance is left running.

If the page does not open after 10 minutes:

1. Check that the instance has a **Public IPv4 address** and that you typed `http://` (not `https://`) and `:8501`.
2. EC2 → Instances → select it → **Actions → Monitor and troubleshoot → Get system log**. The end of the log shows the boot script's steps (Docker install, `git clone`, `docker build`) and the first error, if any.
3. Still unclear: in `stockwise-sg` add an inbound rule *SSH*, port 22, source = the prefix list `com.amazonaws.eu-central-1.ec2-instance-connect` (AWS's own Instance Connect addresses; no key pair needed). Then select the instance → **Connect → EC2 Instance Connect** and run `sudo tail -50 /var/log/cloud-init-output.log` and `sudo docker ps`. Remove the rule afterwards.

### Update the results

After re-running the pipeline (`src/run_pipeline.py` → `src/forecast_best_model.py` → `src/inventory.py`):

```bash
bash deploy/refresh_data.sh
```

Commit `deploy/data/`, merge to `main`, then launch a new server from the template. It always builds from the latest `main`.

### Next step for production

Push the image to **Amazon ECR** and run it on **ECS Fargate** behind a load balancer with HTTPS: no server to manage, and it restarts by itself.

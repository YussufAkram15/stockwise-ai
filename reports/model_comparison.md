# StockWise AI — Model Comparison Report

## 1. Experiment Overview

- Training end: 2017-06-30
- Validation start: 2017-07-01
- Validation end: 2017-08-15
- Number of features: 38
- Validation rows: 81,972
- Best model: **XGBoost**

## 2. Model Performance

| Model | RMSLE | Training Time (seconds) |
|---|---:|---:|
| XGBoost **(Best)** | 0.40216 | 43.4 |
| RandomForest | 0.98715 | 61.5 |
| LinearRegression | 1.39075 | 14.0 |

## 3. Best Model

The best model is **XGBoost**, with a validation RMSLE of **0.40216**.

## 4. Feature Importance

Feature importance values are model-specific. The comparison chart normalizes each model's values separately for visual comparison; this does not make the underlying importance methods directly equivalent.

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | sales_lag_7 | 0.443045 |
| 2 | sales_roll_7_mean | 0.408862 |
| 3 | sales_roll_14_mean | 0.046752 |
| 4 | sales_roll_30_mean | 0.031579 |
| 5 | quarter | 0.010399 |
| 6 | sales_lag_14 | 0.008492 |
| 7 | sales_roll_7_std | 0.007952 |
| 8 | is_payday | 0.004532 |
| 9 | sales_lag_16 | 0.004003 |
| 10 | days_to_month_end | 0.003903 |
| 11 | month | 0.003012 |
| 12 | is_weekend | 0.002756 |
| 13 | day_of_week | 0.002630 |
| 14 | sales_lag_30 | 0.002186 |
| 15 | oil_roll_7_mean | 0.001937 |
| 16 | oil_is_filled | 0.001929 |
| 17 | oil_price | 0.001777 |
| 18 | day_of_month | 0.001723 |
| 19 | store_family_mean_sales | 0.001576 |
| 20 | onpromotion | 0.001479 |

## 5. Generated Charts

| Chart | Description |
|---|---|
| 01 | RMSLE comparison |
| 02 | Training speed vs accuracy |
| 03 | Best model feature importance |
| 04 | Normalized feature importance comparison |
| 05 | Actual vs predicted sales |
| 06 | Log-error distribution |
| 07 | RMSLE by product family |
| 08 | Daily actual vs predicted sales |

Charts are saved in `reports/figures/`.

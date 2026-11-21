# StockWise AI - Feature report

Forecast horizon: 16 days. Sales-history features are shifted by 16 days.
Run `python src/leakage_check.py` separately to test for leakage.

## Size
- Rows: 3,000,888; columns: 51
- Training-period rows (<= 2017-06-30): 2,918,916
- Rows between 2017-06-30 and 2017-07-30: 53,460

## Feature groups

| Group | Features |
|---|---|
| Calendar | `day_of_week`, `day_of_month`, `week_of_year`, `month`, `quarter`, `year`, `is_weekend`, `days_to_month_end`, `is_payday`, `days_to_next_holiday`, `days_since_last_holiday` |
| Sales history | `sales_lag_16`, `sales_lag_21`, `sales_lag_28`, `sales_roll_7_mean`, `sales_roll_28_mean` |
| Promotion / oil | `promo_lag_7`, `promo_roll_7_mean`, `oil_lag_16`, `oil_lag_21`, `oil_lag_28` |
| Level / labels | `store_family_mean_sales`, `store_type_enc`, `family_enc`, `city_enc`, `state_enc` |

## Missing values

| Feature | Missing rows |
|---|---:|
| `sales_lag_16` | 28,512 |
| `sales_lag_21` | 37,422 |
| `sales_lag_28` | 49,896 |
| `sales_roll_7_mean` | 28,512 |
| `sales_roll_28_mean` | 28,512 |
| `promo_lag_7` | 12,474 |
| `promo_roll_7_mean` | 1,782 |
| `oil_lag_16` | 28,512 |
| `oil_lag_21` | 37,422 |
| `oil_lag_28` | 49,896 |
| `store_family_mean_sales` | 28,512 |

## QC flags in this dataset
- `qc_zero_series = 1`: 90,936 rows (series never sold)
- `qc_before_first_sale = 1`: 553,761 rows (before first sale)
- `qc_sales_suspect = 1`: 1,954 rows (suspected sales data errors)

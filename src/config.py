"""Central configuration for the StockWise AI project.

Defines project paths, data-cleaning rules, validation settings,
feature schemas, and model artifact locations.
"""

from __future__ import annotations

from pathlib import Path

# ----------------------------------------------------------------------------
# Paths (relative to the project root, so it works on any computer)
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# All reports and comparison outputs
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"





RAW_FILES = {
    "train": RAW_DIR / "train.csv",
    "test": RAW_DIR / "test.csv",
    "stores": RAW_DIR / "stores.csv",
    "oil": RAW_DIR / "oil.csv",
    "holidays": RAW_DIR / "holidays_events.csv",
    "transactions": RAW_DIR / "transactions.csv",
    "sample_submission": RAW_DIR / "sample_submission.csv",
}
REQUIRED_RAW = ("train", "stores", "oil", "holidays")  # test is optional

TRAIN_CLEAN_PATH = PROCESSED_DIR / "train_clean.parquet"
TEST_CLEAN_PATH = PROCESSED_DIR / "test_clean.parquet"
HOLIDAYS_CLEAN_PATH = PROCESSED_DIR / "holidays_clean.csv"
OIL_DAILY_PATH = PROCESSED_DIR / "oil_daily_filled.csv"
SERIES_SUMMARY_PATH = PROCESSED_DIR / "series_summary.csv"

FEATURES_TRAIN_PATH = PROCESSED_DIR / "train_features.parquet"
FEATURES_TEST_PATH = PROCESSED_DIR / "test_features.parquet"

MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"

EDA_DIR = REPORTS_DIR / "eda"
INSPECTION_REPORT_PATH = REPORTS_DIR / "inspection_report.txt"
CLEANING_REPORT_PATH = REPORTS_DIR / "cleaning_report.md"
VALIDATION_REPORT_PATH = REPORTS_DIR / "validation_report.txt"
FEATURE_REPORT_PATH = REPORTS_DIR / "feature_report.md"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.md"
BEST_PREDICTIONS_PATH = REPORTS_DIR / "best_model_predictions.parquet"

MODEL_COMPARISON_RESULTS_PATH = (
    REPORTS_DIR / "model_comparison_results.parquet"
)

MODEL_COMPARISON_METADATA_PATH = (
    REPORTS_DIR / "model_comparison_metadata.json"
)

# ----------------------------------------------------------------------------
# Cleaning decisions (change them here, and only here)
# ----------------------------------------------------------------------------
# Holiday rows that are REAL days off (when transferred == False).
REAL_HOLIDAY_TYPES = ("Holiday", "Transfer", "Additional", "Bridge")
EVENT_TYPE = "Event"          # Black Friday, Mother's Day, earthquake ... (not a day off)
WORK_DAY_TYPE = "Work Day"    # a Saturday that is a normal working day (not a holiday)
KNOWN_HOLIDAY_TYPES = REAL_HOLIDAY_TYPES + (EVENT_TYPE, WORK_DAY_TYPE)
KNOWN_LOCALES = ("National", "Regional", "Local")

# Quality-control statistics (zero series, first sale, outlier limits) use ONLY
# data up to this date, so nothing from validation/test leaks into the flags.
# Must be the same date as "train ends" in Sara's split.
CLEANING_STATS_CUTOFF = "2017-06-30"

# ----------------------------------------------------------------------------
# Train / validation split (Sara's split — time-based, never shuffle)
# ----------------------------------------------------------------------------
TRAIN_END   = "2017-06-30"   # last day of the training window
VAL_START   = "2017-07-01"   # first day of the validation window
VAL_END     = "2017-07-30"   # last day of the validation window (30 days)
TEST_START  = "2017-07-31"   # first day of the test window (16 days = the forecast horizon)
TEST_END    = "2017-08-15"   # last day of the test window (= last day of train.csv)
# The test window is scored ONCE, at the end, for the chosen model only.

# Outlier rule: a sale is flagged when it is above  Q3 + k * IQR  of the
# non-zero days of ITS OWN series (store x family). Flagged, never removed.
OUTLIER_IQR_MULT = 3.0
OUTLIER_MIN_NONZERO_DAYS = 30   # shorter series get no outlier limit

# ----------------------------------------------------------------------------
# What the official Kaggle "Favorita" files look like (used as WARN checks)
# ----------------------------------------------------------------------------
EXPECTED = {
    "train_rows": 3_000_888,
    "n_stores": 54,
    "n_families": 33,
    "n_train_dates": 1684,          # 1688 calendar days minus 4 x Dec 25
    "train_start": "2013-01-01",
    "train_end": "2017-08-15",
    "test_start": "2017-08-16",
    "test_end": "2017-08-31",
    "test_rows": 28_512,
}

# ----------------------------------------------------------------------------
# Output schema (column order of the cleaned files)
# ----------------------------------------------------------------------------
CALENDAR_FLAGS = [
    "is_holiday", "is_holiday_national", "is_holiday_regional", "is_holiday_local",
    "is_event", "is_earthquake", "is_work_day",
]
QC_FLAGS = [
    "qc_zero_series", "qc_before_first_sale", "qc_store_closed_day", "qc_sales_outlier",
]
TRAIN_CLEAN_COLUMNS = (
    ["id", "date", "store_nbr", "family", "sales", "onpromotion",
     "city", "state", "store_type", "cluster",
     "oil_price", "oil_is_filled"]
    + CALENDAR_FLAGS + QC_FLAGS
)
TEST_CLEAN_COLUMNS = [
    c for c in TRAIN_CLEAN_COLUMNS if c != "sales" and c not in QC_FLAGS
]

COLUMN_DOC = {
    "id": "Row id from the original file (kept so we can always trace a row back).",
    "date": "Day of the sale.",
    "store_nbr": "Store number (1-54).",
    "family": "Product family (33 families).",
    "sales": "Units sold that day by that store for that family. NOT modified by cleaning.",
    "onpromotion": "Number of items of that family on promotion that day in that store.",
    "city": "Store city (from stores.csv).",
    "state": "Store state (from stores.csv).",
    "store_type": "Store type A-E (the 'type' column of stores.csv, renamed to avoid clashing with holiday 'type').",
    "cluster": "Store cluster number (from stores.csv).",
    "oil_price": "Daily oil price. Missing days carried forward from the last known price.",
    "oil_is_filled": "1 if that day had no real oil price in oil.csv and the value was filled.",
    "is_holiday": "1 if a REAL holiday applies to this store on this day (national OR regional OR local).",
    "is_holiday_national": "1 if a real national holiday (all stores).",
    "is_holiday_regional": "1 if a real holiday of the store's state.",
    "is_holiday_local": "1 if a real holiday of the store's city.",
    "is_event": "1 if an event row applies (Black Friday, Mother's Day, earthquake ...). Not a day off.",
    "is_earthquake": "1 on the April 2016 earthquake event days.",
    "is_work_day": "1 on a 'Work Day' (a day that is normally off but worked to pay back a bridge day).",
    "qc_zero_series": "QC FLAG: the whole store x family series has zero sales up to the cutoff date.",
    "qc_before_first_sale": "QC FLAG: day is before the first sale of that series (not opened / not sold yet), up to the cutoff.",
    "qc_store_closed_day": "QC FLAG: the store sold nothing in ANY family that day (closed or not open yet).",
    "qc_sales_outlier": "QC FLAG: sale is above Q3 + k*IQR of its own series (limit learned before the cutoff).",
}
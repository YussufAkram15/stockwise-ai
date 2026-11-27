#!/usr/bin/env bash
# Copy the 6 result files the dashboard reads from reports/ into deploy/data/.
# Run from the project root after:  src/run_pipeline.py -> src/forecast_best_model.py -> src/inventory.py
set -euo pipefail
cd "$(dirname "$0")/.."

FILES="forecast_16d.parquet recent_sales.parquet val_predictions.parquet reorder_table.csv model_comparison.csv test_result.json"
mkdir -p deploy/data
for f in $FILES; do
    cp "reports/$f" "deploy/data/$f"
    echo "copied $f"
done
echo "Done. Commit deploy/data/ so the server gets the new results."

# Labeling + Modeling Pipeline

Quickstart
- Labeling (baseline + with5m):
  - `python -m pipeline.source.labeling.run_labeling`
- Training (default uses with5m labeled CSV):
  - `python -m pipeline.source.labeling.run_training`

Switch between baseline and with5m
- `--label-source baseline` or `--label-source with5m`
- Or pass a direct path: `--input-csv path/to/labeled.csv`

Enable high precision mode (label cleaning)
- Labeling: `python -m pipeline.source.labeling.run_labeling --clean --min-range-pct 0.002`
- Training: `python -m pipeline.source.labeling.run_training --clean --min-range-pct 0.002`

Interpret high precision mode
- Cleaning forces ambiguous candles to `skip` and optionally removes small-range candles.
- Expect higher precision on `long`/`short` with lower coverage (fewer trades).

Outputs
- `pipeline/artifacts/models/`: trained models (joblib)
- `pipeline/artifacts/reports/`: metrics, predictions, thresholds, confusion matrices
- `pipeline/artifacts/labeling/`: cleaned labeled CSVs (if enabled)

Dependencies
- `scikit-learn` for splits/metrics/imputation
- `lightgbm`, `xgboost` (and optionally `catboost`) for model training

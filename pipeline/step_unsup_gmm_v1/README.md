# Unsupervised GMM Pipeline v1

This module sweeps Gaussian Mixture Model (GMM) configurations, evaluates them across walk-forward folds,
selects top-K models, exports per-candle cluster probabilities, labels candles using the required labeling
script, and produces an Excel summary of long/short/skip counts.

## Requirements
Install dependencies:

```bash
pip install -r pipeline/step_unsup_gmm_v1/requirements.txt
```

## Configuration
Edit the sweep configuration:

`pipeline/step_unsup_gmm_v1/configs/gmm_sweep.yaml`

## CLI
Run the full pipeline:

```bash
python -m pipeline.step_unsup_gmm_v1.src.cli all --config pipeline/step_unsup_gmm_v1/configs/gmm_sweep.yaml
```

Run individual steps:

```bash
python -m pipeline.step_unsup_gmm_v1.src.cli sweep --config pipeline/step_unsup_gmm_v1/configs/gmm_sweep.yaml
python -m pipeline.step_unsup_gmm_v1.src.cli export-top --config pipeline/step_unsup_gmm_v1/configs/gmm_sweep.yaml
python -m pipeline.step_unsup_gmm_v1.src.cli label-top --config pipeline/step_unsup_gmm_v1/configs/gmm_sweep.yaml
python -m pipeline.step_unsup_gmm_v1.src.cli excel-summary --config pipeline/step_unsup_gmm_v1/configs/gmm_sweep.yaml
```

## Outputs
All artifacts are written to `pipeline/step_unsup_gmm_v1/results` by default:

- `runs/` per-run JSON artifacts
- `ledger/` aggregated tables and top-K selection
- `top_models/` per-candle probabilities for top models
- `labeled/` per-candle labeled CSVs
- `reports/` markdown report and Excel summary

## Labeling Script
The pipeline invokes the required script:
`D:/GitHub/bitcoin-probabilistic-learning/scripts/signals_code_hour_version_1_0.py`

This wrapper script is safe to import and exposes `label_dataframe()` / `label_csv()` for pipeline use.

# BGMM Groups Walk-Forward (Daily BTC Indicators)

This pipeline is the Bayesian Gaussian Mixture Model (BGMM) variant of the
GMM walk-forward group evaluation. It preserves the same splits, grouping,
stacking, preprocessing, metrics, and ranking logic.

## Outputs
Written to:
`pipeline/step_bgmm_groups_walkforward/results`

Files:
- `metrics_tied.csv`
- `metrics_full.csv`
- `metrics_all.csv`
- `top_configs.csv`

## How to run
From repo root:
```bash
python pipeline/step_bgmm_groups_walkforward/run_bgmm_groups_walkforward.py
```

## Notes
- BGMM uses the same `K` sweep (2..10) and covariance types (`tied`, `full`).
- Metrics are identical to the GMM pipeline; BIC/AIC are computed from the
  model log-likelihood and parameter counts when not available on the estimator.


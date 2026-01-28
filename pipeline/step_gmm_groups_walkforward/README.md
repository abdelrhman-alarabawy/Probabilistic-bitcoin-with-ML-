# GMM Groups Walk-Forward (Daily BTC Indicators)

This pipeline evaluates Gaussian Mixture Models (GMM) on BTC daily indicators using
walk-forward splits, feature-group ablations, and multi-run stability checks.

## What it does
- **Walk-forward splits** from 2020-01-01 to 2025-12-31 with fixed train windows of
  2/3/4 years and 1-year tests, stepping forward by 1 year.
- **Grouping** based on TRAIN data only:
  - `corr`: greedy correlation grouping into 5–10 feature groups.
  - `domain`: name-based groups (IV/options, momentum, volatility, liquidity, trend).
  - `stack`: start from the best single group (tied cov) and add the next best groups
    up to 4 total groups.
- **GMM sweep** over K=2..10 with `covariance_type` in {`tied`, `full`}.
- **Stability checks** across seeds [0..4].

## Outputs
All outputs are written to:
`pipeline/step_gmm_groups_walkforward/results`

Required files:
- `metrics_tied.csv`
- `metrics_full.csv`
- `metrics_all.csv`
- `top_configs.csv`

Each row is one aggregated config: (split × group × K × covariance_type),
with multi-run stability metrics included.

## Metrics notes
- **AvgLogLik** is average per-sample log-likelihood (`mean(score_samples)`).
- **BIC/AIC** are computed on both TRAIN and TEST using sklearn `.bic(X)` / `.aic(X)`.
- **Silhouette/DB** use labels from max responsibility. If a single cluster appears,
  the metric is left as NaN.
- **Responsibility entropy** is averaged per sample:
  `H_n = -Σ_i r_ni log(r_ni + eps)`.
- **Multi-run aggregation**:
  - `MultiRun_LL_Std`: std of Test_AvgLogLik across seeds.
  - `MultiRun_Weight_Std_Mean`: mean std of mixture weights (aligned).
  - `MultiRun_MeanShift_Std_Mean`: mean std of L2 mean shifts (aligned).
- **Best-seed metrics**: for each config we select the seed with highest Test_AvgLogLik
  and report its metrics; stability is always computed across all seeds.

## How to run
From repo root:
```bash
python pipeline/step_gmm_groups_walkforward/run_gmm_groups_walkforward.py
```

## Interpretation
Promote configurations with:
1) High **Test_AvgLogLik**
2) Low **Train_BIC** (relative)
3) Low **MultiRun_LL_Std**
4) Non-extreme **Test_RespEntropy** (avoid near-collapsed responsibilities)

`top_configs.csv` applies these rules per split/covariance type, excluding extreme
entropy rows unless no alternatives exist.

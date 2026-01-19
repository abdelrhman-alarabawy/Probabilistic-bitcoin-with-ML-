# Regime/Cluster Experts (12h, v3)

This module adds unsupervised regime discovery and per-regime expert models on top of the 12h (v3) setup while preserving strict no-leakage rules (shift=1).

## How to run

From the repo root:

```bash
python pipeline/step_regime_experts_12h_v3/main.py
```

Key settings live in `pipeline/step_regime_experts_12h_v3/main.py` under `CONFIG`.

## Outputs

Created under `pipeline/step_regime_experts_12h_v3/`:

- `reports/`
  - `data_profile.txt`: rows, columns, date range, missingness
  - `clustering_report.md`: GMM/HMM selection summary and feature lists
  - `cv_report_mode1_walkforward.md`
  - `cv_report_mode2_expanding.md`
  - `cv_report_mode3_rolling.md`
- `results/`
  - `fold_metrics_mode1_walkforward.csv`
  - `fold_metrics_mode2_expanding.csv`
  - `fold_metrics_mode3_rolling_W6.csv`
  - `fold_metrics_mode3_rolling_W7.csv`
  - `overall_summary.csv`
  - `confusion_matrices/*.csv`
- `artifacts/`
  - `cluster_assignments_full.csv` (timestamp, cluster_id, confidence, regime3)
  - `final_cluster_model.joblib`
  - `final_cluster_preprocessor.joblib`
  - `final_model_preprocessor.joblib`
  - `final_global_model.joblib`
  - `final_expert_models/`
- `figures/`
  - `bic_aic_vs_k.png`
  - `silhouette_vs_k.png`
  - `embedding_2d.png`
  - `clusters_over_time.png`
  - `cluster_feature_means.png`
  - `label_distribution_by_cluster.png`

## Leakage prevention

Inside each fold:
- All engineered features are computed from historical data, then shifted by 1 so row t uses info up to t-1.
- Imputer + scaler are fit on TRAIN only.
- GMM/HMM are fit on TRAIN only.
- Cluster assignment for TRAIN and TEST uses the trained clustering model only.
- Expert models train on TRAIN rows only.
- TEST predictions are routed by cluster to the matching expert.

## Regimes and routing

- GMM is the primary clustering method (K selected by BIC; silhouette reported).
- HMM is attempted if `hmmlearn` is installed; otherwise it is skipped.
- Clusters are mapped to 3 regimes (low/mid/high) using TRAIN-only mean realized volatility (fallback to range).
- Small clusters (< `min_cluster_samples`) are routed to the global model unless the merge strategy is enabled.

## Evaluation modes

- Mode1 walk-forward: train=18m, test=6m, step=3m
- Mode2 expanding: train months 1-2 ? test 3; train 1-3 ? test 4; ...
- Mode3 rolling: W in {6,7} months ? test next 2 months; step=2 months

## How outputs map to decisions

- `overall_summary.csv` aggregates metrics by split type and approach (global vs experts) and notes the best macro-F1 per split.
- Confusion matrices are saved per fold and approach.
- Trade-proxy metrics use next-period return and apply `fee_per_trade` for long/short predictions.

# Regime/Cluster Experts (12h, v3)

This module adds regime discovery and expert models on top of the existing 12h anomaly labeling pipeline output. It uses the same dataset and strict no-leakage rules.

## How to run

From the repo root:

```bash
python pipeline/step_regime_experts_12h_v3/main.py
```

Key settings live in `pipeline/step_regime_experts_12h_v3/main.py` under `CONFIG` (seed, fold sizes, K range, small-cluster strategy, anomaly baseline path).

## Outputs

Created under `pipeline/step_regime_experts_12h_v3/`:

- `reports/`
  - `data_profile.txt`: row counts, label distribution, missingness summary
  - `clustering_report.md`: K selection metrics, feature lists, HMM status
  - `cv_report_expanding.md`: expanding-window CV summary
  - `cv_report_rolling.md`: rolling-window CV summary
- `artifacts/`
  - `scaler.joblib`: clustering scaler fit on full data (for reference)
  - `cluster_model.joblib`: final GMM fit on full data (for reference)
  - `cluster_assignments.csv`: timestamp, cluster_id, confidence, regime3
  - `cluster_sizes_*.json`: per-fold cluster sizes and small-cluster handling
  - `cluster_assignments_*.csv`: per-fold cluster assignments
- `results/`
  - `fold_metrics_expanding.csv`: per-fold metrics (expanding CV)
  - `fold_metrics_rolling.csv`: per-fold metrics (rolling CV)
  - `overall_summary.csv`: mean/std metrics by approach and split
  - `confusion_matrices/*.csv`: per-fold confusion matrices
- `figures/`
  - `bic_vs_k.png`
  - `embedding_2d.png`
  - `clusters_over_time.png`
  - `cluster_feature_means.png`

## Leakage prevention

Inside each fold:
- Feature engineering uses only past data, then all model and cluster features are shifted by 1 (t uses up to t-1).
- Scaler and clustering model are fit on train rows only.
- Cluster assignment for train/test uses the trained clustering model only.
- Expert classifiers are trained only on train rows (per cluster).
- Test predictions are routed to the expert for the assigned cluster (or fallback model for small clusters).

No test data is used for K selection, thresholds, or regime mapping inside folds.

## Regimes and mapping

- GMM is the primary clustering method; K is selected by lowest BIC within each fold.
- HMM is attempted if `hmmlearn` is installed; otherwise it is skipped and documented.
- Clusters are mapped to 3 regimes (low/mid/high) by training-only mean realized volatility (fallback to range if needed).

## Expert routing

- Global baseline: one classifier trained on all train rows.
- Regime experts: one classifier per cluster.
- Small clusters (`min_cluster_samples`) are either routed to the global model or merged to the nearest large cluster (configurable).
- If anomaly baseline outputs are present, they are evaluated alongside the models.

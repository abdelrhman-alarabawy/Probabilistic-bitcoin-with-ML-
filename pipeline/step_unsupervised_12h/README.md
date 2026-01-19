# Unsupervised 12h Pipeline

## How to run
- From repo root:
  `python pipeline/step_unsupervised_12h/main.py`

## Optional dependencies
- `hmmlearn` enables the HMM clustering option. If missing, the pipeline falls back to GMM only.
- `lightgbm` or `catboost` are used for expert models when available; otherwise it falls back to scikit-learn.

## Outputs
- `pipeline/step_unsupervised_12h/reports/`
  - `data_profile.txt`: missingness, dtypes, summary stats, time coverage.
  - `clustering_report.md`: feature set, scaler, model selection and metrics.
  - `cv_report_expanding.md`: expanding-window CV summary.
  - `cv_report_rolling.md`: rolling-window CV summary.
- `pipeline/step_unsupervised_12h/artifacts/`
  - `cluster_model.joblib`: clustering model + preprocessor bundle.
  - `scaler.joblib`: clustering scaler.
  - `pca.joblib`: PCA used for the embedding plot.
  - `cluster_assignments.csv`: timestamp, cluster id, confidence, mapped regime.
- `pipeline/step_unsupervised_12h/results/`
  - `fold_metrics_expanding.csv`: per-fold metrics for expanding CV.
  - `fold_metrics_rolling.csv`: per-fold metrics for rolling CV.
  - `overall_summary.csv`: mean/std summary across folds.
  - `confusion_matrices/`: per-fold confusion matrices for each approach.
- `pipeline/step_unsupervised_12h/figures/`
  - `clusters_ts.png`: cluster id over time.
  - `cluster_feature_means.png`: cluster feature means heatmap.
  - `bic_aic_vs_k.png`: model selection curve.
  - `embedding_2d.png`: PCA embedding colored by cluster.
  - `per_cluster_label_distribution.png`: label distribution by cluster.

## Leakage prevention
- Rolling features are computed with `.shift(1)` so they only use past information.
- Each CV fold fits the scaler, imputer, clipping thresholds, and clustering model on the training split only.
- Expert models are trained per cluster using only training rows from that cluster.
- Clusters with too few training samples fall back to the global model for that fold.

## Cluster-to-regime mapping
- If the model finds more than 3 clusters, clusters are mapped into 3 regimes (low/mid/high vol) using the mean of `realized_vol_rolling` and `range_pct`.
- The mapped regime label is saved alongside the raw cluster id in `cluster_assignments.csv`.

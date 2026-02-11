# Step: GMM Groups

This step runs Gaussian Mixture Model (GMM) regime discovery/evaluation on 6 pre-split group datasets (CSV files or one CSV per subfolder).

## Run

From repo root:

```bash
python pipeline/step_gmm_groups/run_gmm_groups.py \
  --root "data/external/1d" \
  --out "pipeline/step_gmm_groups/results" \
  --seeds 1 2 3 4 5
```

Using defaults from `pipeline/step_gmm_groups/config.yaml`:

```bash
python pipeline/step_gmm_groups/run_gmm_groups.py
```

Useful overrides:

```bash
python pipeline/step_gmm_groups/run_gmm_groups.py \
  --feature-mode selected_top10 \
  --selector-method variance_prune \
  --shift 0 \
  --max-walkforward-folds 8
```

## Assumptions implemented

- Input root supports both layouts:
  - `root/*.csv`
  - `root/<group_name>/*.csv`
- Timestamp auto-detected from: `timestamp`, `time`, `date`, `datetime`.
- OHLCV columns (`open/high/low/close/volume`) are excluded from modeling by default.
- Feature shift is configurable and defaults to `0` in this step, as requested.
- Label-like columns matching patterns (`long`, `short`, `skip`, `label`, `target`, `signal`) are excluded.

## What gets produced

For each group:

- `results/<group_name>/selected_features.json`
  - Base candidate feature list and per-fold selected features.
- `results/<group_name>/ledger.csv`
  - One row per run: `(fold, feature-set, K, cov_type, reg_covar, seed)` + metrics + diagnostics flags.
- `results/<group_name>/summary.csv`
  - Aggregated mean/std over seeds for each config, plus success/failure counts.
- `results/<group_name>/top10.csv`
  - Top ranked 10 configurations for that group.
- Optional:
  - `results/<group_name>/diagnostics/*.csv` (hard labels, responsibilities, transition matrices)
  - `results/<group_name>/plots/*.png` (BIC/AIC, entropy, silhouette vs K)

Global file:

- `results/ALL_groups_top10.csv`

## Folds included

- `holdout`: first 80% train, last 20% test (chronological).
- `walkforward`: chronological train/test windows (default 18m/6m step 3m). Falls back to row-based windows if no timestamps.
- `year_based`: train on `[Y0..Yk]`, test on `Yk+1`.

## Feature selection modes

- `selected_top10` (default):
  - `variance_prune` (default): robust variance ranking -> correlation pruning (`Spearman > 0.9`) -> top 10.
  - `pseudo_mi`: pseudo-label GMM + mutual information ranking.
- `all_features`: skip top-10 selection and model all cleaned indicators.

## Ranking rule (Top 10)

Rank-voting on summary metrics:

1. Maximize `test_avg_loglik_mean`
2. Minimize `bic_train_mean`
3. Minimize `avg_entropy_test_mean`

Combined score is a weighted sum of ranks (default equal weights in `config.yaml`).


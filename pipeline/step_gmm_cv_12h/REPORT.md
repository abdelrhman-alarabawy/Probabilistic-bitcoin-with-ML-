# 12h GMM + Rolling CV Report

## Data Summary
- Input: `data\processed\12h_features_indicators_with_ohlcv.csv`
- Rows: 4096
- Columns: 56
- Timestamp column: timestamp

## Labeling
- Base horizon: 60 minutes
- 12h horizon: 720 minutes
- TP points: 2000.0
- SL points: 1000.0

## GMM Regimes
- k selection strategy: global_k_from_first_fold
- Selected k: 2

## Rolling Monthly CV
- Folds: 67
- Best setting: without_regime
- Fold table: `output/folds_table.csv`

## Averaged Metrics
- without_regime: {'accuracy_mean': 0.614954486268799, 'accuracy_std': 0.15534960967389716, 'macro_f1_mean': 0.5015199787036585, 'macro_f1_std': 0.12484277850161124, 'weighted_f1_mean': 0.6190056825200982, 'weighted_f1_std': 0.1559322937732054, 'long_f1_mean': 0.4056106406203117, 'long_f1_std': 0.17922760906787102, 'short_f1_mean': 0.40843765747888733, 'short_f1_std': 0.1967631866998295, 'skip_f1_mean': 0.6905116380117764, 'skip_f1_std': 0.1793237752978126}
- with_regime: {'accuracy_mean': 0.614954486268799, 'accuracy_std': 0.15534960967389716, 'macro_f1_mean': 0.5015199787036585, 'macro_f1_std': 0.12484277850161124, 'weighted_f1_mean': 0.6190056825200982, 'weighted_f1_std': 0.1559322937732054, 'long_f1_mean': 0.4056106406203117, 'long_f1_std': 0.17922760906787102, 'short_f1_mean': 0.40843765747888733, 'short_f1_std': 0.1967631866998295, 'skip_f1_mean': 0.6905116380117764, 'skip_f1_std': 0.1793237752978126}

## Outputs
- Labeled full dataset: `output/labeled_full.csv`
- Regimes full dataset: `output/regimes_full.csv`
- Fold metrics: `output/folds_metrics.csv`

## Plots
- `plots/close_by_regime.png`
- `plots/regime_counts.png`
- `plots/gmm_cluster_stats.png`
- `plots/label_distribution_by_regime.png`
- `plots/confusion_matrix_fold_*.png`

## Notes
- Rolling monthly CV uses strict chronological splits.
- Preprocessing (imputer/scaler) is fit on each training fold only.
- GMM fit is per fold; k is selected using BIC on the earliest training window.
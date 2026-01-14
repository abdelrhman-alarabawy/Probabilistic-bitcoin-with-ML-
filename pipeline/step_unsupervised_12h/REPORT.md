# 12h Regime Discovery Report

## Data Summary
- Input: `data\processed\12h_features_indicators_with_ohlcv.csv`
- Rows: 4096
- Columns: 58
- Timestamp column: timestamp

## HMM Selection
- Selected n_states: 8 (hmmlearn)
- Candidate metrics: `output/hmm_candidates.csv`

## Regime Mapping
- Final regimes: 2
- Mapping method: k-means on state stats (volatility-ordered labels when possible).
- HMM state stats: `output/hmm_state_stats.csv`
- Target regimes: 3; selected: 2

## Labeling (12h adaptation)
- Base horizon: 60 minutes
- 12h horizon: 720 minutes
- TP points: 2000.0
- SL points: 1000.0

## Per-Regime Label Distribution
- regime_0_stable: {'skip': 0.6781609195402298, 'long': 0.165321594521888, 'short': 0.15651748593788212}
- regime_1_extreme: {'short': 0.7142857142857143, 'skip': 0.2857142857142857}

## Outputs
- Combined dataset with states: `output/combined_with_states.csv`
- Combined dataset with labels: `output/combined_with_states_labeled.csv`
- Regime splits: `output/regime_*.csv`
- Labeled regime files: `output/regime_*_labeled.csv`

## Plots
- `plots/price_by_regime.png`
- `plots/state_occupancy.png`
- `plots/hmm_transition_heatmap.png`
- `plots/final_regime_transition_heatmap.png`
- `plots/regime_boxplots.png`

## Notes
- Chronological splits used for HMM training/validation and per-regime modeling.
- If optional ML libraries are missing, fallback models are used.
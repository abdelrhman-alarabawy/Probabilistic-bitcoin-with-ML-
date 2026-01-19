# Regime-First Pipeline Report (12h, v3)

## Data
- Rows used: 2404
- Features used: 76
- Label column: candle_type
- Excluded columns: timestamp, candle_type, open, high, low, close, volume, label_ambiguous

## Fold Metrics (OOS)
- Fold 0: coverage=0.000, precision=0.000, trades=0, eligible_regimes=1, HMM_K=3, GMM_K=3
- Fold 1: coverage=0.000, precision=0.000, trades=0, eligible_regimes=0, HMM_K=3, GMM_K=5
- Fold 2: coverage=0.000, precision=0.000, trades=0, eligible_regimes=0, HMM_K=2, GMM_K=4
- Fold 3: coverage=0.000, precision=0.000, trades=0, eligible_regimes=0, HMM_K=2, GMM_K=4

## Label Distribution by Fold
- Fold 0 train: {'skip': 741, 'long': 253, 'short': 208}
- Fold 0 test: {'skip': 142, 'short': 52, 'long': 46}
- Fold 1 train: {'skip': 883, 'long': 299, 'short': 260}
- Fold 1 test: {'skip': 150, 'short': 54, 'long': 36}
- Fold 2 train: {'skip': 1033, 'long': 335, 'short': 314}
- Fold 2 test: {'skip': 152, 'long': 46, 'short': 42}
- Fold 3 train: {'skip': 1185, 'long': 381, 'short': 356}
- Fold 3 test: {'skip': 167, 'long': 41, 'short': 32}

## Purity Diagnostics (HMM)
- purity_actionable mean=0.541, p50=0.547, p90=0.566, max=0.576
## Purity Diagnostics (GMM)
- purity_actionable mean=0.565, p50=0.564, p90=0.600
## HMM vs GMM Label Skew
- Fold 0: HMM purity_mean=0.553, GMM purity_mean=0.571 -> GMM
- Fold 1: HMM purity_mean=0.537, GMM purity_mean=0.560 -> GMM
- Fold 2: HMM purity_mean=0.536, GMM purity_mean=0.578 -> GMM
- Fold 3: HMM purity_mean=0.534, GMM purity_mean=0.554 -> GMM

## Summary
- Coverage @ 0.95: 0.000
- Precision @ 0.95: 0.000
- Brier score (long vs short): nan

## Frontier (Pareto)
- cov=0.000, prec=0.000, action_rate=0.03, purity=0.5, margin=0.0, entropy_max=inf, decision_thr=0.8
- cov=0.000, prec=0.000, action_rate=0.03, purity=0.5, margin=0.0, entropy_max=inf, decision_thr=0.9
- cov=0.000, prec=0.000, action_rate=0.03, purity=0.5, margin=0.0, entropy_max=inf, decision_thr=0.95
- cov=0.000, prec=0.000, action_rate=0.03, purity=0.5, margin=0.0, entropy_max=0.8, decision_thr=0.8
- cov=0.000, prec=0.000, action_rate=0.03, purity=0.5, margin=0.0, entropy_max=0.8, decision_thr=0.9
- cov=0.000, prec=0.000, action_rate=0.03, purity=0.5, margin=0.0, entropy_max=0.8, decision_thr=0.95
- cov=0.000, prec=0.000, action_rate=0.03, purity=0.5, margin=0.0, entropy_max=0.7, decision_thr=0.8
- cov=0.000, prec=0.000, action_rate=0.03, purity=0.5, margin=0.0, entropy_max=0.7, decision_thr=0.9
- cov=0.000, prec=0.000, action_rate=0.03, purity=0.5, margin=0.0, entropy_max=0.7, decision_thr=0.95
- cov=0.000, prec=0.000, action_rate=0.03, purity=0.5, margin=0.0, entropy_max=0.6, decision_thr=0.8

## Interpretation
- Labels are used as-is; skip is ignored for direction training.

## Next Steps
- Inspect label skew by regime and adjust features for directional separation.
- Consider regime-specific thresholds or alternative state models if purity remains low.
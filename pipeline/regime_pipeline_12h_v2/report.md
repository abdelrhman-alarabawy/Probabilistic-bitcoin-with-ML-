# Regime-First Pipeline Report (12h, v2)

## Data
- Rows used: 2404
- Features used: 75
- Label column: candle_type
- Excluded columns: timestamp, candle_type, open, high, low, close, volume, label_ambiguous

## Fold Metrics (OOS)
- Fold 0: coverage=0.000, precision=0.000, trades=0, eligible_regimes=0, HMM_K=3, GMM_K=6
- Fold 1: coverage=0.000, precision=0.000, trades=0, eligible_regimes=0, HMM_K=3, GMM_K=5
- Fold 2: coverage=0.000, precision=0.000, trades=0, eligible_regimes=0, HMM_K=2, GMM_K=4
- Fold 3: coverage=0.000, precision=0.000, trades=0, eligible_regimes=0, HMM_K=2, GMM_K=4

## Summary
- Coverage @ 0.95: 0.000
- Precision @ 0.95: 0.000

## Eligibility Sweep (No Trades Case)
Relaxed thresholds to show trade-off between coverage and precision.
- action_rate=0.05, purity=0.5: coverage=0.831, precision=0.197, trades=798.0
- action_rate=0.08, purity=0.5: coverage=0.831, precision=0.197, trades=798.0
- action_rate=0.1, purity=0.5: coverage=0.831, precision=0.197, trades=798.0
- action_rate=0.12, purity=0.5: coverage=0.831, precision=0.197, trades=798.0
- action_rate=0.05, purity=0.55: coverage=0.000, precision=0.000, trades=0.0
- action_rate=0.05, purity=0.6: coverage=0.000, precision=0.000, trades=0.0
- action_rate=0.05, purity=0.65: coverage=0.000, precision=0.000, trades=0.0
- action_rate=0.08, purity=0.55: coverage=0.000, precision=0.000, trades=0.0

## Notes
- Labels are used as-is; skip is excluded from direction training.
- Regime eligibility uses label distribution and stability metrics.

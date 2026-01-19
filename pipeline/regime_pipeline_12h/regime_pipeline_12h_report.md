# Regime-First, Uncertainty-Aware Pipeline Report (12h)

## Data Summary
- Rows: 2434
- Features: 53
- Excluded columns: candle_type, label_ambiguous, timestamp
- Suspicious columns: label_ambiguous

## Fold Metrics (OOS)
- Fold 0: coverage=0.000, precision=0.000, trades=0, eligible_regimes=0, HMM_K=4, GMM_K=6
- Fold 1: coverage=0.000, precision=0.000, trades=0, eligible_regimes=0, HMM_K=3, GMM_K=6
- Fold 2: coverage=0.000, precision=0.000, trades=0, eligible_regimes=0, HMM_K=2, GMM_K=2
- Fold 3: coverage=0.000, precision=0.000, trades=0, eligible_regimes=0, HMM_K=2, GMM_K=5

## Strict Mode Table
- threshold=0.8: coverage=0.000, precision=0.000, trades=0
- threshold=0.9: coverage=0.000, precision=0.000, trades=0
- threshold=0.95: coverage=0.000, precision=0.000, trades=0
- threshold=0.98: coverage=0.000, precision=0.000, trades=0

## Conclusions
- Precision is prioritized via regime gating, entropy filters, and calibrated probabilities.
- Coverage is intentionally low at higher thresholds.

## Next Steps
- Consider TP/SL-based labels for more trade-like supervision.
- Try different horizons and eligibility thresholds for higher precision.
- Prune features or add regime-specific thresholds if precision remains low.
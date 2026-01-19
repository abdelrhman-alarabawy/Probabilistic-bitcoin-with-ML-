# Regime-First Pipeline Report (12h, v4)

## Data
- Rows used: 2404
- Features used: 76
- Label column: candle_type
- Excluded columns: timestamp, candle_type, open, high, low, close, volume, label_ambiguous

## Gate Metrics (OOS)
- Fold 0: precision=0.000, recall=0.000, coverage=0.000, trades=0
- Fold 1: precision=0.000, recall=0.000, coverage=0.000, trades=0
- Fold 2: precision=0.000, recall=0.000, coverage=0.000, trades=0
- Fold 3: precision=0.000, recall=0.000, coverage=0.000, trades=0

## Direction Metrics (OOS)
- Fold 0: precision=0.000, coverage=0.000, trades=0
- Fold 1: precision=0.000, coverage=0.000, trades=0
- Fold 2: precision=0.000, coverage=0.000, trades=0
- Fold 3: precision=0.000, coverage=0.000, trades=0

## Summary
- Coverage @ gate=0.9, dir=0.6: 0.000
- Precision (direction): 0.000
- Gate AP (PR curve): 0.353
- Gate Brier score: 0.2372
- P(trade) stats: min=0.000, p50=0.409, max=0.621
- P(direction) stats: min=0.500, p50=0.540, max=0.642

## Debug Modes (Gate Impact)
- gate_only: coverage=0.000, precision_dir=0.000, precision_gate=0.000, trades=0
- gate_entropy: coverage=0.000, precision_dir=0.000, precision_gate=0.000, trades=0
- full: coverage=0.000, precision_dir=0.000, precision_gate=0.000, trades=0

## Frontier (Pareto)
- Mode gate_only:
  - cov=0.034, prec_dir=0.273, prec_gate=0.330, gate_thr=0.425, dir_thr=0.55, entropy_max=inf
  - cov=0.034, prec_dir=0.273, prec_gate=0.330, gate_thr=0.425, dir_thr=0.55, entropy_max=0.8
  - cov=0.034, prec_dir=0.273, prec_gate=0.330, gate_thr=0.425, dir_thr=0.55, entropy_max=0.7
  - cov=0.034, prec_dir=0.273, prec_gate=0.330, gate_thr=0.425, dir_thr=0.55, entropy_max=0.6
  - cov=0.075, prec_dir=0.222, prec_gate=0.341, gate_thr=0.424, dir_thr=0.55, entropy_max=inf
- Mode full:
  - no non-zero coverage points

## Interpretation
- Direction thresholds are intentionally lower because regime purity is near coin-flip.
- Gate model drives precision by filtering out skip-heavy periods.
- If max P(trade) is below the gate threshold grid, coverage will be zero.
- Dynamic gate thresholds now use per-fold P(trade) quantiles and top-k modes.
- Compressed gate scores came from calibration on imbalanced labels; isotonic expands range but still limited.

## Next Steps
- Use frontier_full.csv + frontier_pareto.csv to choose gate/dir thresholds.
- Consider regime-specific gate thresholds if trade coverage remains low.
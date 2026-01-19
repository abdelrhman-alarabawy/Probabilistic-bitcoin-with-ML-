# Regime-First Rolling Pipeline Report (12h, v7)

## Data
- Rows used: 3993
- Features used: 116
- Label column: candle_type
- Excluded columns: timestamp, candle_type, open, high, low, close, volume, label_ambiguous

## Gate Confusion Summary
| K | Median precision | Median recall | Median FPR | Std precision | Aggregate precision |
| --- | --- | --- | --- | --- | --- |
| 5 | 0.400 | 0.014 | 0.013 | 0.199 | 0.386 |
| 10 | 0.300 | 0.033 | 0.026 | 0.174 | 0.364 |
| 20 | 0.350 | 0.059 | 0.053 | 0.119 | 0.343 |

## Baseline Comparison
| K | Gate median precision | Random-K mean precision | Volatility top-K mean precision | Always-trade precision |
| --- | --- | --- | --- | --- |
| 5 | 0.400 | 0.306 | 0.329 | 0.302 |
| 10 | 0.300 | 0.297 | 0.343 | 0.302 |
| 20 | 0.350 | 0.302 | 0.332 | 0.302 |

## Stability and Best K
- Best K by median precision: 5 (median precision=0.400).
- Fraction of windows beating random-K precision: K=5: 0.50, K=10: 0.57, K=20: 0.50.

## Conclusion
- Gate evaluation uses top-K trades per window to guarantee trade counts.
- See outputs/gate_confusion_by_window.csv and outputs/baselines_by_window.csv for details.
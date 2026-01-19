# Regime-First Rolling Pipeline Report (12h, v5)

## Data
- Rows used: 2404
- Features used: 83
- Label column: candle_type
- Excluded columns: timestamp, candle_type, open, high, low, close, volume, label_ambiguous

## Window Setup
- Default window: 18m train / 6m test / 3m step to increase trades per test window.
- Optional comparison: 18m train / 3m test / 3m step.

## Window Summary (by config)
| Window | Median gate AP | Median precision<=5% | % windows w/ trades | Median trades | Stability (std) |
| --- | --- | --- | --- | --- | --- |
| train18_test6_step3 | 0.370 | 0.333 | 0.00 | 4.0 | 0.142 |

## Gate Calibration Diagnostics
- Median gate AP: 0.370
- Median gate Brier: 0.240

## Window Quality Timeline
- See figures: `figures/window_quality_*.png` (GOOD/BAD/INSUFFICIENT).

## Frontier Interpretation
- Best p10 policy: topk topk_2 dir_thr=0.55 entropy_max=0.6 (p10=0.073, median=0.167, coverage=0.046).

## Conclusion
- Directional signal at 12h remains weak; use the label primarily as a trade-timing signal.
- Gate ranking provides the most stable signal under non-stationarity.
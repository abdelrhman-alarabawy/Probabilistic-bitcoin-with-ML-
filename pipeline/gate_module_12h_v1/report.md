# Gate Module Report (12h, v1)

## Data
- Rows used: 3993
- Features used: 116
- Label column: candle_type
- Excluded columns: timestamp, candle_type, open, high, low, close, volume, label_ambiguous

## Gate Summary (K sweep)
| K | Median precision | Aggregate precision | Median FPR | Windows beating random | Always-trade precision (action rate) | Random-K mean precision | Random-K precision std |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 0.200 | 0.271 | 0.016 | 0.36 | 0.302 | 0.306 | 0.203 |
| 10 | 0.300 | 0.336 | 0.028 | 0.50 | 0.302 | 0.297 | 0.141 |
| 20 | 0.375 | 0.357 | 0.049 | 0.64 | 0.302 | 0.302 | 0.098 |

## Trade vs Not-Trade Confusion (Top-K=20)

| Actual \ Predicted | Trade | Notrade |
| --- | --- | --- |
| Trade | 100 | 1442 |
| Notrade | 180 | 3382 |

- Aggregate precision: 0.357, recall: 0.065, FPR: 0.051.
- Total used: 5104, total excluded: 4 (raw test rows=5108).
- Exclusions by reason: timestamp=0, missing label=0, missing features=4, other=0.
- N_scored matches N_cm_used for all windows.
- Top missing features: hurst_proxy_100 (n=101), max_drawdown_100 (n=100), autocorr_return_50_lag2 (n=51), autocorr_return_50_lag1 (n=51), rolling_kurt_50 (n=51), rolling_skew_50 (n=51), cvar_proxy_50 (n=51), downside_vol_50 (n=51), return_z_50 (n=51), bb_bandwidth_50 (n=50).

## Conclusion
- Gate provides a weak-but-real timing filter.
- Do not use gate scores to predict direction.
- Recommended production setting: TOP-K=20.
- K can change with dataset/windowing because base rates and window mix shift; selection is data-driven.
# Regime-First Rolling Pipeline Report (12h, v6)

## Data
- Rows used: 3993
- Features used: 116
- Label column: candle_type
- Excluded columns: timestamp, candle_type, open, high, low, close, volume, label_ambiguous

## Feature Engineering
- Added features: 95
- Final features after pruning: 116
- Catalog saved to outputs/features_added_catalog.csv

## Regime Separability
- Selected method counts: core=12, pca=2
- Diagnostics saved to outputs/regime_separability.csv

## Window Summary (by config)
| Window | Median gate AP | Median precision<=5% | % windows w/ trades | Median trades | Stability (std) |
| --- | --- | --- | --- | --- | --- |
| train18_test6_step3 | 0.361 | 0.278 | 0.00 | 1.5 | 0.279 |

## Gate Calibration Diagnostics
- Median gate AP: 0.361
- Median gate Brier: 0.218

## Comparison vs v5
- v5 comparison unavailable.

## Frontier Interpretation
- Best p10 policy: threshold q90 dir_thr=0.58 entropy_max=0.7 (p10=0.000, median=0.172, coverage=0.047).

## Conclusion
- Directional signal at 12h remains weak; improvements mainly target gate ranking and regime separation.
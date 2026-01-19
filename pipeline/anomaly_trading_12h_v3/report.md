# Anomaly Trading Report (12h, v3)

## Setup
- Data: 12h_features_indicators_with_ohlcv.csv
- Features shift: t uses data up to t-1 (shift=1)
- Windows: train=18m, test=6m, step=3m
- Fee per trade: 0.0005
- Bucket4 mode: skip

## Percentile Sweep (dedup, pre-filters)
| Variant | Percentile | Signals | Signals/mo |
| --- | --- | --- | --- |
| no_liq | 95 | 115 | 2.35 |
| no_liq | 96 | 84 | 1.71 |
| no_liq | 97 | 62 | 1.27 |
| no_liq | 98 | 33 | 0.67 |
| no_liq | 99 | 16 | 0.33 |
| with_liq | 95 | 73 | 1.49 |
| with_liq | 96 | 45 | 0.92 |
| with_liq | 97 | 35 | 0.71 |
| with_liq | 98 | 28 | 0.57 |
| with_liq | 99 | 10 | 0.20 |

## Filter Audit
| Variant | EMA200 | Signals | After liq | After bucket | After EMA | Executed |
| --- | --- | --- | --- | --- | --- | --- |
| no_liq | False | 81 | 81 | 1 | 1 | 1 |
| no_liq | True | 81 | 81 | 1 | 1 | 1 |
| with_liq | False | 41 | 20 | 2 | 2 | 2 |
| with_liq | True | 41 | 20 | 2 | 2 | 2 |

## Best Configs (under constraints)
| Rank | Variant | EMA200 | Hold | TP | SL | Trades | Signals/mo | Win rate | Avg ret | CVaR95 | Max DD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | with_liq | False | 1 | 0.0050 | 0.0050 | 2 | 0.04 | 0.000 | -0.0055 | -0.0055 | -0.0055 |
| 2 | with_liq | False | 1 | 0.0075 | 0.0050 | 2 | 0.04 | 0.000 | -0.0055 | -0.0055 | -0.0055 |
| 3 | with_liq | False | 1 | 0.0100 | 0.0050 | 2 | 0.04 | 0.000 | -0.0055 | -0.0055 | -0.0055 |

## Comparison Summary
| Variant | EMA200 | Best win rate | Trades | Signals/mo |
| --- | --- | --- | --- | --- |
| no_liq | False | 0.000 | 1 | 0.02 |
| no_liq | True | 0.000 | 1 | 0.02 |
| with_liq | False | 0.000 | 2 | 0.04 |
| with_liq | True | 0.000 | 2 | 0.04 |

## Figures
- figures/equity_top1.png
- figures/signals_timeline_top1.png
- figures/return_dist_top1.png

## Notes
- RobustZ only; MOM strategy removed.
- Adaptive thresholds by range_strength buckets.
- Dedup keeps earliest window decision per timestamp.
# Interval Prediction Report (12h, v1)

## Setup
- Horizon (bars): 1
- Windows: train=18m, test=6m, step=3m
- Features used: 116 (from gate_module_12h_v1)
- Quantile pairs: [(0.1, 0.9), (0.2, 0.8), (0.3, 0.7)]
- Width thresholds: [0.005, 0.01, 0.02]
- Gate enabled: True

## Interval Quality Summary (Aggregate)
| Target | q_low | q_high | Gate | Coverage | Width% mean | Width% median |
| --- | --- | --- | --- | --- | --- | --- |
| high | 0.10 | 0.90 | all | 0.332 | 0.099 | 0.072 |
| high | 0.10 | 0.90 | gate | 0.379 | 0.100 | 0.076 |
| high | 0.20 | 0.80 | all | 0.256 | 0.077 | 0.055 |
| high | 0.20 | 0.80 | gate | 0.311 | 0.067 | 0.051 |
| high | 0.30 | 0.70 | all | 0.185 | 0.056 | 0.037 |
| high | 0.30 | 0.70 | gate | 0.225 | 0.052 | 0.036 |
| low | 0.10 | 0.90 | all | 0.317 | 0.096 | 0.068 |
| low | 0.10 | 0.90 | gate | 0.371 | 0.107 | 0.095 |
| low | 0.20 | 0.80 | all | 0.243 | 0.071 | 0.056 |
| low | 0.20 | 0.80 | gate | 0.268 | 0.072 | 0.061 |
| low | 0.30 | 0.70 | all | 0.149 | 0.048 | 0.030 |
| low | 0.30 | 0.70 | gate | 0.179 | 0.050 | 0.033 |

## Tight Precision (Aggregate)
| Target | q_low | q_high | Gate | Prec@W0.005 | Prec@W0.010 | Prec@W0.020 |
| --- | --- | --- | --- | --- | --- | --- |
| high | 0.10 | 0.90 | all | 0.007 | 0.019 | 0.030 |
| high | 0.10 | 0.90 | gate | 0.000 | 0.000 | 0.024 |
| high | 0.20 | 0.80 | all | 0.013 | 0.012 | 0.039 |
| high | 0.20 | 0.80 | gate | 0.000 | 0.000 | 0.014 |
| high | 0.30 | 0.70 | all | 0.013 | 0.023 | 0.041 |
| high | 0.30 | 0.70 | gate | 0.000 | 0.023 | 0.022 |
| low | 0.10 | 0.90 | all | 0.003 | 0.007 | 0.019 |
| low | 0.10 | 0.90 | gate | 0.000 | 0.000 | 0.030 |
| low | 0.20 | 0.80 | all | 0.003 | 0.017 | 0.025 |
| low | 0.20 | 0.80 | gate | 0.000 | 0.000 | 0.018 |
| low | 0.30 | 0.70 | all | 0.013 | 0.017 | 0.030 |
| low | 0.30 | 0.70 | gate | 0.029 | 0.019 | 0.030 |

## High vs Low Comparison
- High coverage (median): 0.283; Low coverage (median): 0.255.
- High width% mean (median): 0.072; Low width% mean (median): 0.071.

## Gate vs No-Gate Comparison
- high: coverage median gate=0.311, all=0.256.
- low: coverage median gate=0.268, all=0.243.

## Best High-Precision Regimes
- high: W=0.005: q=(0.30,0.70), all (prec=0.013); W=0.010: q=(0.30,0.70), all (prec=0.023); W=0.020: q=(0.30,0.70), all (prec=0.041).
- low: W=0.005: q=(0.30,0.70), gate (prec=0.029); W=0.010: q=(0.30,0.70), gate (prec=0.019); W=0.020: q=(0.30,0.70), all (prec=0.030).

## Notes
- Top missing features: hurst_proxy_100 (n=101), max_drawdown_100 (n=100), autocorr_return_50_lag2 (n=51), autocorr_return_50_lag1 (n=51), rolling_kurt_50 (n=51), rolling_skew_50 (n=51), cvar_proxy_50 (n=51), downside_vol_50 (n=51), return_z_50 (n=51), bb_bandwidth_50 (n=50).

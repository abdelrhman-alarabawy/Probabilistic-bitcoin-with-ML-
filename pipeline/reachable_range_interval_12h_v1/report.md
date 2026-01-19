# Reachable Range Interval Report (12h, v1)

## Setup
- Reach horizon (bars): 3
- Windows: train=18m, test=6m, step=3m
- Features used: 116 (from gate_module_12h_v1)
- Quantile pairs: [(0.1, 0.9), (0.2, 0.8), (0.3, 0.7)]
- Width thresholds: [0.005, 0.01, 0.02]
- Gate enabled: True
- Rows: raw=4096, after_clean=4096, after_targets=4093, after_features=3990.

## Aggregate Coverage + Width
| Target | q_low | q_high | Gate | Coverage | Width% mean | Width% median | Gap vs nominal |
| --- | --- | --- | --- | --- | --- | --- | --- |
| high_reach | 0.10 | 0.90 | all | 0.307 | 0.119 | 0.088 | -0.493 |
| high_reach | 0.10 | 0.90 | gate | 0.346 | 0.122 | 0.092 | -0.454 |
| high_reach | 0.20 | 0.80 | all | 0.251 | 0.085 | 0.063 | -0.349 |
| high_reach | 0.20 | 0.80 | gate | 0.254 | 0.083 | 0.065 | -0.346 |
| high_reach | 0.30 | 0.70 | all | 0.186 | 0.067 | 0.042 | -0.214 |
| high_reach | 0.30 | 0.70 | gate | 0.214 | 0.060 | 0.047 | -0.186 |
| low_reach | 0.10 | 0.90 | all | 0.306 | 0.118 | 0.088 | -0.494 |
| low_reach | 0.10 | 0.90 | gate | 0.354 | 0.123 | 0.091 | -0.446 |
| low_reach | 0.20 | 0.80 | all | 0.204 | 0.072 | 0.050 | -0.396 |
| low_reach | 0.20 | 0.80 | gate | 0.254 | 0.077 | 0.059 | -0.346 |
| low_reach | 0.30 | 0.70 | all | 0.153 | 0.053 | 0.037 | -0.247 |
| low_reach | 0.30 | 0.70 | gate | 0.186 | 0.056 | 0.045 | -0.214 |

## Tight Precision (Aggregate)
| Target | q_low | q_high | Gate | Prec@W0.005 | Prec@W0.010 | Prec@W0.020 |
| --- | --- | --- | --- | --- | --- | --- |
| high_reach | 0.10 | 0.90 | all | 0.016 | 0.020 | 0.021 |
| high_reach | 0.10 | 0.90 | gate | 0.000 | 0.038 | 0.040 |
| high_reach | 0.20 | 0.80 | all | 0.000 | 0.013 | 0.029 |
| high_reach | 0.20 | 0.80 | gate | 0.000 | 0.000 | 0.000 |
| high_reach | 0.30 | 0.70 | all | 0.007 | 0.015 | 0.029 |
| high_reach | 0.30 | 0.70 | gate | 0.000 | 0.000 | 0.038 |
| low_reach | 0.10 | 0.90 | all | 0.016 | 0.007 | 0.016 |
| low_reach | 0.10 | 0.90 | gate | 0.100 | 0.056 | 0.050 |
| low_reach | 0.20 | 0.80 | all | 0.003 | 0.007 | 0.018 |
| low_reach | 0.20 | 0.80 | gate | 0.000 | 0.000 | 0.000 |
| low_reach | 0.30 | 0.70 | all | 0.009 | 0.015 | 0.029 |
| low_reach | 0.30 | 0.70 | gate | 0.000 | 0.000 | 0.000 |

## Best High-Precision Regimes
- high_reach: W=0.005: q=(0.30,0.70), all (prec=0.007, cov_tight=0.079); W=0.010: q=(0.10,0.90), gate (prec=0.038, cov_tight=0.093); W=0.020: q=(0.10,0.90), gate (prec=0.040, cov_tight=0.179).
- low_reach: W=0.005: q=(0.30,0.70), all (prec=0.009, cov_tight=0.088); W=0.010: q=(0.10,0.90), gate (prec=0.056, cov_tight=0.064); W=0.020: q=(0.10,0.90), gate (prec=0.050, cov_tight=0.143).

## Commentary
- Reachable range targets reduce path noise but do not guarantee high tight precision.
- Compare tight-precision tables vs the original next-candle interval report.

## Notes
- Exclusions: timestamp/duplicates=0, target tail=3, missing features=103.
- Top missing features: hurst_proxy_100 (n=101), max_drawdown_100 (n=100), autocorr_return_50_lag2 (n=51), autocorr_return_50_lag1 (n=51), rolling_kurt_50 (n=51), rolling_skew_50 (n=51), cvar_proxy_50 (n=51), downside_vol_50 (n=51), return_z_50 (n=51), bb_bandwidth_50 (n=50).

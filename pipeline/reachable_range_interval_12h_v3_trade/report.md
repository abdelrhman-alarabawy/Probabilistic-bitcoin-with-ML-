# Reachable Range Trade Report (12h, v3)

## Setup
- Reach horizon (bars): 3
- Coverage target: 0.70
- LIQ_POLICY: drop_all_liq
- Gate enabled: True
- Features used: 116
- Volatility feature: atr_pct_14 (k=1.00)
- Rows: raw=4096, after_clean=4096, after_targets=4093.

## Trade Rules
- intersection: width_high<=thr_high AND width_low<=thr_low
- either: width_high<=thr_high OR width_low<=thr_low
- vol_adjusted: width_high<=thr_high*(1+k*vol) AND width_low<=thr_low*(1+k*vol)

## Interval Coverage + Width (per target)
| Target | Gate | Coverage | Width% mean | Width% median |
| --- | --- | --- | --- | --- |
| high_reach | all | 0.709 | 0.044 | 0.040 |
| high_reach | gate | 0.687 | 0.044 | 0.040 |
| low_reach | all | 0.727 | 0.047 | 0.042 |
| low_reach | gate | 0.710 | 0.044 | 0.044 |

## Tightness Thresholds (median over windows)
| Target | Threshold | Median value |
| --- | --- | --- |
| high_reach | p01 | 0.0296 |
| high_reach | p05 | 0.0329 |
| high_reach | p10 | 0.0345 |
| high_reach | p25 | 0.0381 |
| low_reach | p01 | 0.0266 |
| low_reach | p05 | 0.0293 |
| low_reach | p10 | 0.0310 |
| low_reach | p25 | 0.0332 |

## Tight Precision (intervals)
| Target | Gate | p01 | p05 | p10 | p25 |
| --- | --- | --- | --- | --- | --- |
| high_reach | all | 0.516 | 0.653 | 0.631 | 0.663 |
| high_reach | gate | 0.200 | 0.427 | 0.434 | 0.635 |
| low_reach | all | 0.531 | 0.596 | 0.685 | 0.668 |
| low_reach | gate | 0.117 | 0.133 | 0.330 | 0.515 |

## Trade Confusion (Aggregate)
| Rule | Thr | Gate | Precision | Recall | FPR | Trades |
| --- | --- | --- | --- | --- | --- | --- |
| vol_adjusted | p25 | gate | 0.391 | 0.098 | 0.067 | 23 |
| vol_adjusted | p10 | gate | 0.333 | 0.022 | 0.019 | 6 |
| vol_adjusted | p05 | gate | 0.333 | 0.011 | 0.010 | 3 |
| either | p25 | gate | 0.321 | 0.478 | 0.447 | 137 |
| intersection | p25 | gate | 0.294 | 0.054 | 0.058 | 17 |
| either | p25 | all | 0.278 | 0.416 | 0.467 | 2473 |
| either | p10 | gate | 0.269 | 0.196 | 0.236 | 67 |
| either | p10 | all | 0.264 | 0.191 | 0.230 | 1197 |
| vol_adjusted | p25 | all | 0.261 | 0.094 | 0.115 | 593 |
| intersection | p10 | gate | 0.250 | 0.011 | 0.014 | 4 |
| either | p01 | all | 0.244 | 0.042 | 0.057 | 287 |
| intersection | p25 | all | 0.243 | 0.064 | 0.087 | 437 |
| either | p05 | all | 0.239 | 0.110 | 0.152 | 763 |
| vol_adjusted | p10 | all | 0.181 | 0.018 | 0.034 | 160 |
| either | p05 | gate | 0.171 | 0.076 | 0.163 | 41 |
| either | p01 | gate | 0.133 | 0.022 | 0.062 | 15 |
| vol_adjusted | p05 | all | 0.128 | 0.006 | 0.018 | 78 |
| intersection | p10 | all | 0.120 | 0.008 | 0.025 | 108 |
| intersection | p01 | all | 0.111 | 0.001 | 0.004 | 18 |
| intersection | p05 | all | 0.088 | 0.003 | 0.014 | 57 |
| vol_adjusted | p01 | all | 0.087 | 0.001 | 0.005 | 23 |
| intersection | p01 | gate | 0.000 | 0.000 | 0.005 | 1 |
| vol_adjusted | p01 | gate | 0.000 | 0.000 | 0.005 | 1 |
| intersection | p05 | gate | 0.000 | 0.000 | 0.005 | 1 |

## Recommendation
- Best rule under MIN_TRADES=5: either @ p25 (all) (precision=0.278, recall=0.416).

## Gate Impact
- Best precision gate=0.391, all=0.278; gate helps precision.

## Notes
- actual_trade = candle_type in {long, short}; skip = no-trade.
- Top missing features: hurst_proxy_100 (n=101), max_drawdown_100 (n=100), autocorr_return_50_lag2 (n=51), autocorr_return_50_lag1 (n=51), rolling_kurt_50 (n=51), rolling_skew_50 (n=51), cvar_proxy_50 (n=51), downside_vol_50 (n=51), return_z_50 (n=51), bb_bandwidth_50 (n=50).
- Dropped liq features: 10 columns.

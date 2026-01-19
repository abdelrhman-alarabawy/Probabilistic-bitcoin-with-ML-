# Reachable Range Interval Report (12h, v2 CQR)

## Setup
- Reach horizon (bars): 3
- Windows: train=18m, test=6m, step=3m
- Features used: 116 (from gate_module_12h_v1)
- Coverage target: 0.70 (alpha_low=0.15, alpha_high=0.85)
- Gate enabled: True
- Rows: raw=4096, after_clean=4096, after_targets=4093.

## Aggregate Coverage + Width
| Target | Gate | Coverage | Width% mean | Width% median | Width price mean | Gap vs nominal |
| --- | --- | --- | --- | --- | --- | --- |
| high_reach | all | 0.709 | 0.044 | 0.041 | 2182.116 | 0.009 |
| high_reach | gate | 0.687 | 0.044 | 0.041 | 2277.094 | -0.013 |
| low_reach | all | 0.727 | 0.047 | 0.044 | 2236.212 | 0.027 |
| low_reach | gate | 0.710 | 0.044 | 0.043 | 2343.022 | 0.010 |

## Tight Precision
| Target | Gate | Prec@p10 | Prec@p25 | Prec@0.005 | Prec@0.010 | Prec@0.020 |
| --- | --- | --- | --- | --- | --- | --- |
| high_reach | all | 0.628 | 0.655 | 0.000 | 0.000 | 0.400 |
| high_reach | gate | 0.633 | 0.640 | 0.000 | 0.000 | 0.000 |
| low_reach | all | 0.640 | 0.674 | 0.000 | 0.000 | 0.714 |
| low_reach | gate | 0.567 | 0.600 | 0.000 | 0.000 | 0.667 |

## Best High-Precision Regimes
- high_reach: p10: gate (prec=0.633); p25: all (prec=0.655).
- low_reach: p10: all (prec=0.640); p25: all (prec=0.674).

## Commentary
- Reachable range targets smooth path noise; CQR calibration targets coverage but tight precision may remain low.

## Notes
- Exclusions: timestamp/duplicates=0, target tail=3.
- Top missing features: liq_buy_count (n=1659), liq_count (n=1659), liq_net_notional_long_minus_short (n=1659), liq_sell_notional (n=1659), liq_sell_amount (n=1659), liq_buy_notional (n=1659), liq_buy_amount (n=1659), liq_sell_count (n=1659), liq_notional (n=1659), liq_amount (n=1659).

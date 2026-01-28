# Regime validation report

## Configuration
- input_csv: D:\GitHub\bitcoin-probabilistic-learning\data\processed\12h_features_indicators_with_ohlcv.csv
- preds_csv: D:\GitHub\bitcoin-probabilistic-learning\pipeline\step_regime_gmm_12h_open\results\gmm_regimes_per_row.csv
- TP_POINTS: 2000
- SL_POINTS: 1000
- fee_per_trade: 0.0005
- thresholds: [0.8, 0.9, 0.95, 0.98, 0.99]

## Top tradable regimes (ranked)
| regime | n | n_trades | profit_factor_all | avg_pnl_all | decision_tradable | decision_side |
| --- | --- | --- | --- | --- | --- | --- |
| 7 | 452 | 85 | inf | 0.019726 | 1 | none |
| 3 | 134 | 52 | inf | 0.019720 | 0 | none |
| 2 | 797 | 268 | inf | 0.019707 | 1 | both |
| 6 | 874 | 376 | inf | 0.019703 | 1 | both |
| 4 | 280 | 106 | inf | 0.019700 | 0 | both |
| 5 | 1040 | 271 | inf | 0.019685 | 1 | both |
| 0 | 505 | 160 | inf | 0.019681 | 1 | both |
| 1 | 13 | 3 | inf | 0.019636 | 0 | none |

## Do trade
| regime | decision_side | profit_factor_all | avg_pnl_all |
| --- | --- | --- | --- |
| 0 | both | inf | 0.019681 |
| 2 | both | inf | 0.019707 |
| 5 | both | inf | 0.019685 |
| 6 | both | inf | 0.019703 |
| 7 | none | inf | 0.019726 |

## Don't trade
| regime | decision_side | profit_factor_all | avg_pnl_all |
| --- | --- | --- | --- |
| 1 | none | inf | 0.019636 |
| 3 | none | inf | 0.019720 |
| 4 | both | inf | 0.019700 |

## Best threshold per regime
| regime | best_threshold | profit_factor | n_trades |
| --- | --- | --- | --- |
| 0.000000 | 0.800000 | inf | 160.000000 |
| 1.000000 | 0.800000 | inf | 3.000000 |
| 2.000000 | 0.800000 | inf | 266.000000 |
| 3.000000 | 0.800000 | inf | 51.000000 |
| 4.000000 | 0.800000 | inf | 106.000000 |
| 5.000000 | 0.800000 | inf | 271.000000 |
| 6.000000 | 0.800000 | inf | 374.000000 |
| 7.000000 | 0.800000 | inf | 84.000000 |

## Warnings
| regime | n | decision_regime_type |
| --- | --- | --- |
| 1 | 13 | transition_noise |
| 3 | 134 | transition_noise |
| 4 | 280 | transition_noise |

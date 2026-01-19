# Anomaly Trading Report (12h)

## Setup
- Data: 12h_features_indicators_with_ohlcv.csv
- Features shift: t uses data up to t-1 (shift=1)
- Windows: train=18m, test=6m, step=3m
- Hold bars: 3, TP=0.010, SL=0.010
- MOM: near_high>=0.80, near_low<=0.20, range_expand: prev_range > median*1.00 (lookback=50)

## Feature Set
- Base features (11): return_1, return_2, return_3, range_pct, atr_14_pct, volatility_14, downside_vol_14, drawdown_lookback, zscore_return_lookback, volume_zscore, volume_zscore_missing
- Liquidity features (10): liq_count, liq_amount, liq_notional, liq_buy_count, liq_sell_count, liq_buy_amount, liq_buy_notional, liq_sell_amount, liq_sell_notional, liq_net_notional_long_minus_short
- Missingness flags: on

## Thresholds (Median over windows)
| Model | Percentile | Median threshold |
| --- | --- | --- |
| iforest | 98 | 0.6293 |
| iforest | 99 | 0.6715 |
| robustz | 98 | 25.8280 |
| robustz | 99 | 33.9480 |

## Strategy Metrics (Aggregate)
| Model | Pct | Strategy | Trades | Signals/mo | Win rate | Avg ret | Med ret | MAE | MFE | Max DD | CVaR95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| iforest | 98 | MOM | 52 | 0.72 | 0.167 | -0.0062 | -0.0100 | -0.0271 | 0.0168 | -0.0960 | -0.0083 |
| iforest | 99 | MOM | 8 | 0.22 | 0.222 | -0.0052 | -0.0100 | -0.0270 | 0.0193 | -0.0136 | -0.0067 |
| robustz | 98 | MOM | 26 | 0.43 | 0.033 | -0.0093 | -0.0100 | -0.0312 | 0.0231 | -0.0491 | -0.0100 |
| robustz | 99 | MOM | 12 | 0.33 | 0.097 | -0.0081 | -0.0100 | -0.0248 | 0.0184 | -0.0199 | -0.0100 |
| iforest | 98 | MR | 228 | 2.71 | 0.486 | -0.0006 | -0.0050 | -0.0254 | 0.0276 | -0.1232 | -0.0071 |
| iforest | 99 | MR | 105 | 1.75 | 0.432 | -0.0018 | -0.0050 | -0.0170 | 0.0279 | -0.0959 | -0.0080 |
| robustz | 98 | MR | 58 | 0.88 | 0.425 | -0.0015 | -0.0000 | -0.0219 | 0.0289 | -0.0491 | -0.0082 |
| robustz | 99 | MR | 24 | 0.44 | 0.509 | 0.0002 | 0.0000 | -0.0274 | 0.0260 | -0.0101 | -0.0078 |

## Stability Across Windows
| Model | Pct | Strategy | Win rate std | Avg return std |
| --- | --- | --- | --- | --- |
| iforest | 98 | MOM | 0.290 | 0.0058 |
| iforest | 99 | MOM | 0.404 | 0.0082 |
| robustz | 98 | MOM | 0.071 | 0.0014 |
| robustz | 99 | MOM | 0.153 | 0.0031 |
| iforest | 98 | MR | 0.310 | 0.0062 |
| iforest | 99 | MR | 0.297 | 0.0061 |
| robustz | 98 | MR | 0.303 | 0.0061 |
| robustz | 99 | MR | 0.305 | 0.0061 |

## Figures
- figures/score_hist.png
- figures/signals_timeline.png
- figures/equity_curve.png

## Notes
- No labels changed; candle_type is untouched.
- Test windows overlap (step < test length), so trades may repeat across windows.
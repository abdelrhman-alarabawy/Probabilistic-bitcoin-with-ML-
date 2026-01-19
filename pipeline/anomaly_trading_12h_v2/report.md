# Anomaly Trading Report (12h, v2)

## Setup
- Data: 12h_features_indicators_with_ohlcv.csv
- Features shift: t uses data up to t-1 (shift=1)
- Windows: train=18m, test=6m, step=3m
- Fee per trade: 0.0005

## Dedup Summary
- Raw test rows: 5478
- Dedup test rows: 2922
- p98 raw signals: 58, dedup signals: 28, duplicates removed: 30
- p99 raw signals: 24, dedup signals: 10, duplicates removed: 14

## Feature Set
- Base features (11): return_1, return_2, return_3, range_pct, atr_14_pct, volatility_14, downside_vol_14, drawdown_lookback, zscore_return_lookback, volume_zscore, volume_zscore_missing
- Liquidity features (10): liq_count, liq_amount, liq_notional, liq_buy_count, liq_sell_count, liq_buy_amount, liq_buy_notional, liq_sell_amount, liq_sell_notional, liq_net_notional_long_minus_short
- Missingness flags: on

## Best Configs (focus p99)
- p99 total trades after filters: 0
- Rare high precision (p98, win rate max, signals/mo<=2, trades>=15): hold=1, tp=0.0050, sl=0.0050, range_thr=0.0030, win_rate=1.000, avg_ret=0.0045, trades=1, signals/mo=0.02
- Rare high-precision constraints not met; showing best available config.
- Best expectancy (p98, avg return max, trades>=30): hold=1, tp=0.0150, sl=0.0050, range_thr=0.0030, win_rate=1.000, avg_ret=0.0145, trades=1, signals/mo=0.02
- Best-expectancy constraints not met; showing best available config.

## Confusion-Style Summary
- Rare high precision (wins/losses)
  wins=1, losses=0
- Best expectancy (wins/losses)
  wins=1, losses=0

## Figures
- figures/equity_best.png
- figures/signals_best.png
- figures/return_distribution.png

## Notes
- RobustZ only; MOM strategy removed.
- Trend filters: range_strength + EMA200 regime gate.
- Dedup keeps earliest window decision per timestamp.
- p99 produced zero trades after filters; best configs fall back to other percentiles.
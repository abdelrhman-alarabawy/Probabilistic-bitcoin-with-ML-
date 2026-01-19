# Features (v6)

This pipeline adds leakage-safe, regime-focused features computed using only historical data (shifted by 1 where needed).

## Groups

A) Return & volatility structure
- log_return_1 = log(close/close.shift(1)) using shifted close
- realized_vol_{w} = rolling std of log_return_1, w ∈ {5,10,20}
- vol_of_vol_{w} = rolling std of realized_vol_{w}, w ∈ {5,10}
- atr_{w}, atr_pct_{w} with w ∈ {7,14,28}
- parkinson_vol_{w}, gk_vol_{w} with w ∈ {10,20}

B) Trend strength vs chop
- ema_slope_{w} = diff(EMA(close,w))/close, w ∈ {10,20,50}
- ma_cross_{a,b} = EMA(a) - EMA(b), (10,20),(20,50)
- adx_{w}, w ∈ {7,14,28}
- aroon_up/down/osc_{w}, w ∈ {14,28}
- r2_trend_{w} = rolling R^2 of close vs time, w ∈ {20,50}

C) Compression / expansion
- bb_bandwidth_{w}, bb_pctb_{w}, w ∈ {20,50}
- donchian_width_{w}, w ∈ {20,50}
- keltner_width_{w}, w ∈ {20,50}
- squeeze_flag_{w}
- range_pct + rolling mean/std w ∈ {10,20}

D) Candle microstructure
- body_pct, upper_wick_pct, lower_wick_pct, close_location_value, gap_pct
- rolling mean/std of body/wick metrics w ∈ {5,10,20}

E) Volume / liquidity
- vol_log
- volume_z_{w}, w ∈ {20,50}
- obv, obv_slope_{w}, w ∈ {10,20}
- chaikin_mf, cmf_20
- vpt, vpt_slope_{w}, w ∈ {10,20}
- dollar_volume

F) Tail-risk / stress
- return_z_{w}, w ∈ {20,50}
- downside_vol_{w}, w ∈ {20,50}
- cvar_proxy_50
- max_drawdown_{w}, w ∈ {50,100}
- jump_flag

G) Distributional
- rolling_skew_50, rolling_kurt_50
- hurst_proxy_100
- autocorr_return_50_lag1/lag2

See `outputs/features_added_catalog.csv` for full catalog with formulas and lookbacks.

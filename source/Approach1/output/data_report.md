# Approach1 Data Report

## Dataset Overview

Rows: 30614
Timestamp column: `ts_utc`

## Label Balance

- skip: 19779
- short: 5436
- long: 5399

## Random Variables

Continuous RVs: 19
Discrete RVs (native): 1
Discrete RVs (with bins): 18

## Missingness

- aggressor_bias: 71.09%
- liquidity_intensity: 4.63%
- signal_confidence: 0.40%
- signal_direction_score: 0.40%
- momentum_score: 0.40%
- order_imbalance_score: 0.33%
- tail_risk_z: 0.24%
- trend_score: 0.16%
- mean_reversion_score: 0.16%
- realized_vol_24h: 0.08%
- volume_trend_24h: 0.08%
- volume_z_24h: 0.08%
- spread_cost_bps: 0.02%
- iv_level: 0.02%
- term_structure_slope: 0.02%

## Top Correlations (Absolute)

- momentum_score vs signal_direction_score: 1.000
- ret_1h vs body_1h: 1.000
- ret_1h vs tail_risk_z: 0.869
- body_1h vs tail_risk_z: 0.868
- trend_score vs mean_reversion_score: 0.863
- realized_vol_24h vs iv_level: 0.746
- direction_sign vs tail_risk_z: 0.622
- mean_reversion_score vs signal_direction_score: 0.614
- ret_1h vs direction_sign: 0.613
- direction_sign vs body_1h: 0.612

## Leakage Checklist

- Rolling windows use shifted inputs (t-1 and earlier).
- No features use negative shifts or future timestamps.
- Labels are not used to construct features.

## RV Definitions (Summary)

- ret_1h: Log return over 1 hour.
- direction_sign: Signed return direction (-1, 0, 1).
- range_1h: Normalized high-low range.
- body_1h: Normalized candle body.
- realized_vol_24h: Trailing realized volatility (24 bars).
- tail_risk_z: Return z-score versus trailing window (tail risk proxy).
- momentum_score: Average z-scored multi-horizon return momentum.
- trend_score: Distance between fast and slow moving averages.
- mean_reversion_score: Negative distance from slow moving average (overbought/oversold).
- volume_z_24h: Volume z-score versus trailing window.
- volume_trend_24h: Slope of volume over trailing window.
- order_imbalance_score: Order book imbalance z-score (last and mean).
- spread_cost_bps: Estimated spread cost in basis points.
- liquidity_intensity: Log-scaled liquidation intensity.
- aggressor_bias: Net aggressor bias from liquidation flows.
- iv_level: Average ATM implied volatility level.
- term_structure_slope: Implied volatility term structure slope.
- implied_move_ratio: Ratio of 1h to 24h implied move.
- signal_direction_score: Weighted direction score from trend and momentum.
- signal_confidence: Soft confidence score in [0,1] for downstream evidence.

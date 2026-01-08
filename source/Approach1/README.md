# Approach1: Random Variable Construction for BTC 1H

This module defines a compact, interpretable set of random variables (RVs) for the 1-hour BTC dataset and prepares both continuous and discretized versions for later Bayesian Network modeling.

Random variables here mean: **probabilistic quantities derived from observed market data at time t** that can become nodes in a Bayesian Network. Each RV is constructed using only data available at or before time t (no future peeking), and each has a clear trading intuition.

## Inputs and assumptions

- Timestamp column auto-detected from: `nts-utc`, `ts_utc`, `timestamp`, `time`, `datetime`, `date`.
- Label column auto-detected from: `candle_type`, `Candle_type`, `CandleType`, `label`, `y`.
- OHLCV columns are required.
- Optional columns (order book, options, liquidation) are used when present.

## Random variables (continuous)

Each formula uses values at time t and historical values up to t-1. For rolling stats, the inputs are shifted by 1 to avoid look-ahead.

1) **ret_1h**
   - Formula: `log(C_t / C_{t-1})`
   - Inputs: `close`
   - Shift: 1
   - Intuition: Immediate return signal.

2) **range_1h**
   - Formula: `(H_t - L_t) / C_t`
   - Inputs: `high`, `low`, `close`
   - Shift: 0
   - Intuition: Intrabar volatility proxy.

3) **body_1h**
   - Formula: `(C_t - O_t) / C_t`
   - Inputs: `open`, `close`
   - Shift: 0
   - Intuition: Candle conviction (directional body size).

4) **realized_vol_24h**
   - Formula: `std(ret_{t-1..t-24})`
   - Inputs: `ret_1h`
   - Shift: 1
   - Window: 24
   - Intuition: Trailing realized risk.

5) **tail_risk_z**
   - Formula: `(ret_t - mean(ret_{t-1..t-N})) / std(ret_{t-1..t-N})`
   - Inputs: `ret_1h`
   - Shift: 1
   - Window: 72
   - Intuition: Large-move detection.

6) **momentum_score**
   - Formula: `mean(zscore(log(C_t/C_{t-h})))` for h in {3,6,12,24}
   - Inputs: `close`
   - Shift: 1
   - Window: 120 (for z-score)
   - Intuition: Multi-horizon momentum strength.

7) **trend_score**
   - Formula: `(MA_fast - MA_slow) / C_t`
   - Inputs: `close`
   - Shift: 1
   - Windows: 12, 48
   - Intuition: Trend slope between fast and slow averages.

8) **mean_reversion_score**
   - Formula: `-(C_t - MA_slow) / MA_slow`
   - Inputs: `close`
   - Shift: 1
   - Window: 48
   - Intuition: Overbought/oversold pressure.

9) **volume_z_24h**
   - Formula: `(V_t - mean(V_{t-1..t-24})) / std(V_{t-1..t-24})`
   - Inputs: `volume`
   - Shift: 1
   - Window: 24
   - Intuition: Relative volume burst.

10) **volume_trend_24h**
    - Formula: `slope(V_{t-1..t-24})`
    - Inputs: `volume`
    - Shift: 1
    - Window: 24
    - Intuition: Volume up/down trend.

11) **order_imbalance_score** (if available)
    - Formula: `mean(zscore(imb_last), zscore(imb_mean))`
    - Inputs: `imbalance_last`, `imbalance_mean`
    - Shift: 1
    - Window: 24
    - Intuition: Order flow pressure.

12) **spread_cost_bps** (if available)
    - Formula: `(ask_t - bid_t) / mid_t * 1e4`
    - Inputs: `spread_bps_last` or `spread_last` or `bid_last`+`ask_last`
    - Shift: 0
    - Intuition: Liquidity / execution cost proxy.

13) **liquidity_intensity** (if available)
    - Formula: `log(1 + liq_metric_t)`
    - Inputs: `liq_notional` or `liq_amount` or `liq_count`
    - Shift: 0
    - Intuition: Liquidation activity level.

14) **aggressor_bias** (if available)
    - Formula: `(buy_notional - sell_notional) / (buy_notional + sell_notional)`
    - Inputs: `liq_buy_notional`, `liq_sell_notional` or `liq_net_notional_long_minus_short`
    - Shift: 0
    - Intuition: Directional liquidation flow.

15) **iv_level** (if available)
    - Formula: `mean(atm_iv_1d, atm_iv_2d, atm_iv_7d)`
    - Inputs: `atm_iv_*`
    - Shift: 0
    - Intuition: Implied volatility regime.

16) **term_structure_slope** (if available)
    - Formula: `term_slope_1d_7d` or `atm_iv_7d - atm_iv_1d`
    - Inputs: `term_slope_1d_7d` or `atm_iv_*`
    - Shift: 0
    - Intuition: Volatility term structure tilt.

17) **implied_move_ratio** (if available)
    - Formula: `implied_move_1h_pct_1sigma / implied_move_24h_pct_1sigma`
    - Inputs: `implied_move_1h_pct_1sigma`, `implied_move_24h_pct_1sigma`
    - Shift: 0
    - Intuition: Short-term risk pricing vs daily.

18) **signal_direction_score**
    - Formula: `w_m * momentum_score + w_t * trend_score`
    - Inputs: `momentum_score`, `trend_score`
    - Shift: 0
    - Intuition: Directional consensus score.

19) **signal_confidence**
    - Formula: `sigmoid(|signal_dir|) * alignment * exp(-|tail_risk_z|)`
    - Inputs: `signal_direction_score`, `tail_risk_z`
    - Shift: 0
    - Intuition: Soft evidence for BN (0 to 1).

## Random variables (discrete)

- **direction_sign**: `sign(ret_1h)` in {-1, 0, 1}.
- For each continuous RV, a discretized counterpart is created using quantile bins, with suffix `_bin` (default 3 bins). The bin edges are stored in `rv_metadata.json`.

## Leakage checks

- All rolling windows use shifted inputs (t-1 and earlier).
- No feature uses negative shifts.
- Labels are not used to construct RVs.
- Timestamp order enforced and duplicates removed.

## Summary table (candidates for BN nodes)

| RV name | Type | Parent candidates | Expected relationship |
| --- | --- | --- | --- |
| ret_1h | continuous | trend_score, momentum_score, order_imbalance_score, volume_z_24h | Trend or flow aligned moves raise expected return |
| range_1h | continuous | realized_vol_24h, iv_level, spread_cost_bps | Higher risk regimes widen range |
| body_1h | continuous | trend_score, momentum_score, order_imbalance_score | Stronger trend increases candle body |
| realized_vol_24h | continuous | iv_level, term_structure_slope, volume_z_24h | Higher implied vol precedes higher realized vol |
| tail_risk_z | continuous | realized_vol_24h, iv_level | Extreme tail moves in high vol regimes |
| momentum_score | continuous | ret_1h (lags), trend_score | Momentum persists in trending markets |
| trend_score | continuous | momentum_score, iv_level | Longer-term drift increases trend score |
| mean_reversion_score | continuous | trend_score, realized_vol_24h | Overshoot risk rises after strong trend |
| volume_z_24h | continuous | liquidity_intensity, aggressor_bias | Flow shocks increase volume z-score |
| volume_trend_24h | continuous | volume_z_24h, liquidity_intensity | Sustained flow shifts volume slope |
| order_imbalance_score | continuous | volume_z_24h, spread_cost_bps | Imbalance intensifies during active flow |
| spread_cost_bps | continuous | liquidity_intensity, volume_z_24h | Thin liquidity widens spread |
| liquidity_intensity | continuous | tail_risk_z, iv_level | Stress events raise liquidation intensity |
| aggressor_bias | continuous | order_imbalance_score, trend_score | Flow bias aligns with trend |
| iv_level | continuous | realized_vol_24h (lag), term_structure_slope | Implied vol responds to recent risk |
| term_structure_slope | continuous | iv_level, implied_move_ratio | Short-term fear steepens the slope |
| implied_move_ratio | continuous | iv_level, realized_vol_24h | Short-term risk pricing rises in turmoil |
| signal_direction_score | continuous | momentum_score, trend_score | Consensus signal reflects trend + momentum |
| signal_confidence | continuous | signal_direction_score, tail_risk_z | Confidence falls in extreme tail moves |
| direction_sign | discrete | ret_1h | Directional sign of return |
| candle_type | discrete | signal_direction_score, signal_confidence | Strategy label aligned with signal regime |

## How to run

```bash
python -m source.Approach1.src.build_rvs --config source/Approach1/config.yaml
```

### Expected outputs

- `source/Approach1/output/random_variables_continuous.csv`
- `source/Approach1/output/random_variables_discrete.csv`
- `source/Approach1/output/rv_metadata.json`
- `source/Approach1/output/data_report.md`

## Dependencies

- Python 3.10+
- pandas, numpy, pyyaml

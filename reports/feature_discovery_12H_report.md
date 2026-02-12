# 12H Feature Discovery Report

## Input Files
- data\external\12H\ohlcv_with_market_structure_indicators (1).csv
- data\external\12H\ohlcv_with_orderflow_derivatives_indicators (1).csv
- data\external\12H\ohlcv_with_oscillator_indicators (1).csv
- data\external\12H\ohlcv_with_trend_indicators (1).csv
- data\external\12H\ohlcv_with_volatility_indicators (1).csv
- data\external\12H\ohlcv_with_volume_indicators (1).csv

## Merge Summary
- inner_rows: 4094
- outer_rows: 4095
- merge_mode: inner
- duplicate_timestamps_by_file: {'ohlcv_with_market_structure_indicators (1).csv': 0, 'ohlcv_with_orderflow_derivatives_indicators (1).csv': 0, 'ohlcv_with_oscillator_indicators (1).csv': 0, 'ohlcv_with_trend_indicators (1).csv': 0, 'ohlcv_with_volatility_indicators (1).csv': 0, 'ohlcv_with_volume_indicators (1).csv': 0}

## Label Summary
- counts: {'skip': 2774, 'long': 675, 'short': 645}

## Feature Pruning
- initial_features: 340
- corr_threshold: 0.95

## Shift Sweep
- shifts: [1, 2, 3]
- feature_shift means features at t use data from <= t-shift

### Shift 1
- removed_constants: 2
- removed_corr: 66
- missingness_before: 0.0146
- missingness_after: 0.0000
- long pr_auc_valid=0.1368 pr_auc_test=0.1479
- long gated_features=['up_capture_60d', 'avg_loss_20d', 'pvi', 'realized_vol_20d', 'realized_vol_60d']
- short pr_auc_valid=0.2066 pr_auc_test=0.1403
- short gated_features=['atr_7', 'obv_x', 'price_range_pct_14', 'avg_loss_60d', 'parkinson_vol_30', 'avg_gain_60d']
- skip pr_auc_valid=0.6932 pr_auc_test=0.7917
- skip gated_features=['returns_25th_252d', 'obv_x', 'days_since_high', 'cumulative_return', 'realized_vol_60d', 'pvt', 'imbalance_std', 'up_capture_60d']

### Shift 2
- removed_constants: 2
- removed_corr: 66
- missingness_before: 0.0146
- missingness_after: 0.0000
- long pr_auc_valid=0.1565 pr_auc_test=0.1182
- long gated_features=['pvi', 'up_capture_20d']
- short pr_auc_valid=0.1943 pr_auc_test=0.1417
- short gated_features=['imbalance_std', 'returns_kurtosis_252d', 'days_since_high', 'cumulative_return', 'vwap_x', 'realized_vol_20d']
- skip pr_auc_valid=0.7408 pr_auc_test=0.8197
- skip gated_features=['obv_x', 'realized_vol_10d', 'imbalance_std']

### Shift 3
- removed_constants: 2
- removed_corr: 66
- missingness_before: 0.0146
- missingness_after: 0.0000
- long pr_auc_valid=0.1436 pr_auc_test=0.1233
- long gated_features=['realized_vol_60d']
- short pr_auc_valid=0.2122 pr_auc_test=0.1483
- short gated_features=['obv_x', 'cumulative_return', 'avg_gain_60d', 'down_capture_60d']
- skip pr_auc_valid=0.7619 pr_auc_test=0.7883
- skip gated_features=['returns_iqr_60d', 'obv_x', 'volume_momentum_10', 'avg_loss_60d']

## Regime Conditioning (Best Shift Only)
- selected_shift_for_regimes: 3
- regime_features_selected: ['returns_x', 'returns_2d', 'returns_5d', 'returns_10d', 'returns_20d', 'returns_50d', 'returns_skewness_20d', 'returns_skewness_60d', 'returns_skewness_252d', 'returns_kurtosis_20d', 'returns_kurtosis_60d', 'returns_kurtosis_252d']
- regime_K_selected: 4
- regime_0: n=1187
  - long pr_auc_valid=0.2222 pr_auc_test=0.1485
  - short pr_auc_valid=0.2139 pr_auc_test=0.1492
  - skip pr_auc_valid=0.6599 pr_auc_test=0.8060
- regime_1: n=724
  - long pr_auc_valid=0.2784 pr_auc_test=0.2293
  - short pr_auc_valid=0.3804 pr_auc_test=0.2643
  - skip pr_auc_valid=0.6712 pr_auc_test=0.5972
- regime_2: n=1170
  - long pr_auc_valid=0.1437 pr_auc_test=0.2040
  - short pr_auc_valid=0.1533 pr_auc_test=0.1991
  - skip pr_auc_valid=0.8582 pr_auc_test=0.8299
- regime_3: n=1010
  - long pr_auc_valid=0.2502 pr_auc_test=0.1427
  - short pr_auc_valid=0.1081 pr_auc_test=0.0971
  - skip pr_auc_valid=0.7904 pr_auc_test=0.7446
## Recommendations
- best_shift_long: 2
- best_shift_short: 3
- regime_conditioning: review per-regime PR-AUC for gains
- if PR-AUC < 0.30: labels may be noisy, consider horizon adjustment
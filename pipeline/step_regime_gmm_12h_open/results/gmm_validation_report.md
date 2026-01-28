# GMM regime discovery validation report

## Configuration
- input_csv: D:\GitHub\bitcoin-probabilistic-learning\data\processed\12h_features_indicators_with_ohlcv.csv
- mode: fit_full
- shift_periods: 1 (features shifted by 1 for OPEN entry)
- initial_train_months: 18
- rows_used: 4095
- features_used: 38

## Model selection (top 3 by BIC)
| rank | k | covariance | BIC | AIC | avg_max_prob | avg_entropy |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 8 | full | -663881.35 | -703296.37 | 0.9954 | 0.0129 |
| 2 | 6 | full | -663865.32 | -693425.01 | 0.9934 | 0.0165 |
| 3 | 7 | full | -662522.13 | -697009.48 | 0.9940 | 0.0157 |

Selected model: k=8, covariance=full

## Separation diagnostics
- avg max posterior prob: 0.9954
- avg entropy: 0.0129
- pct prob_max >= 0.6: 99.85%
- pct prob_max >= 0.7: 99.58%
- pct prob_max >= 0.8: 99.22%
- pct prob_max >= 0.9: 98.68%

## Regime stability
Transition matrix (counts):

| from | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 490 | 1 | 0 | 9 | 4 | 0 | 1 | 0 |
| 1 | 0 | 2 | 0 | 3 | 7 | 0 | 1 | 0 |
| 2 | 0 | 0 | 751 | 8 | 0 | 2 | 1 | 35 |
| 3 | 11 | 5 | 11 | 55 | 25 | 6 | 21 | 0 |
| 4 | 3 | 2 | 0 | 26 | 95 | 13 | 141 | 0 |
| 5 | 0 | 0 | 3 | 6 | 11 | 973 | 44 | 3 |
| 6 | 1 | 3 | 0 | 24 | 138 | 43 | 665 | 0 |
| 7 | 0 | 0 | 32 | 2 | 0 | 3 | 0 | 414 |

Average run length by regime:

| regime | avg_run_length |
| --- | --- |
| 0 | 33.67 |
| 1 | 1.18 |
| 2 | 17.33 |
| 3 | 1.70 |
| 4 | 1.51 |
| 5 | 15.52 |
| 6 | 4.18 |
| 7 | 11.89 |

## Economic interpretability (next-candle returns)
| regime | count | mean | median | std | sharpe_like | win_rate | q05 | q25 | q50 | q75 | q95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.000000 | 505.000000 | 0.002560 | 0.001014 | 0.020066 | 0.127597 | 0.546535 | -0.024609 | -0.005933 | 0.001014 | 0.011163 | 0.033441 |
| 1.000000 | 13.000000 | 0.011857 | 0.006141 | 0.034224 | 0.346449 | 0.769231 | -0.032083 | 0.000297 | 0.006141 | 0.018284 | 0.071471 |
| 2.000000 | 797.000000 | 0.000959 | 0.001179 | 0.020487 | 0.046794 | 0.539523 | -0.031946 | -0.008915 | 0.001179 | 0.010546 | 0.032812 |
| 3.000000 | 134.000000 | -0.000514 | 0.000898 | 0.034967 | -0.014704 | 0.507463 | -0.055225 | -0.019055 | 0.000898 | 0.021855 | 0.053782 |
| 4.000000 | 280.000000 | 0.002239 | 0.001819 | 0.031932 | 0.070103 | 0.539286 | -0.044251 | -0.011784 | 0.001819 | 0.017124 | 0.046878 |
| 5.000000 | 1040.000000 | 0.001121 | 0.000036 | 0.016731 | 0.067022 | 0.500962 | -0.021495 | -0.005311 | 0.000036 | 0.006195 | 0.028807 |
| 6.000000 | 874.000000 | -0.000196 | 0.000197 | 0.026672 | -0.007332 | 0.502288 | -0.042429 | -0.013335 | 0.000197 | 0.012988 | 0.041999 |
| 7.000000 | 452.000000 | 0.000142 | 0.000231 | 0.012275 | 0.011603 | 0.506637 | -0.021540 | -0.005295 | 0.000231 | 0.006694 | 0.019509 |

## Feature profile (top 15 by abs z-diff)
### Regime 0
| feature | regime_mean | global_mean | z_diff |
| --- | --- | --- | --- |
| local_timestamp_last | 1596271565544117.750000 | 1673536467280996.750000 | -1.513578 |
| implied_move_24h_usd_1sigma | 315.155277 | 1288.578429 | -1.204484 |
| implied_move_4h_usd_1sigma | 128.661603 | 526.059941 | -1.204484 |
| implied_move_1h_usd_1sigma | 64.330802 | 263.029970 | -1.204484 |
| ask_last | 10939.361386 | 47412.906227 | -1.194581 |
| microprice_last | 10939.115060 | 47412.439101 | -1.194572 |
| mid_last | 10939.111386 | 47412.428205 | -1.194572 |
| bid_last | 10938.861386 | 47411.950183 | -1.194563 |
| spot_opt | 10938.004689 | 47462.779993 | -1.194446 |
| imbalance_std | 0.781568 | 0.726517 | 1.001524 |
| fly25_7d | 0.113227 | 0.057483 | 0.961879 |
| spread_std | 0.295140 | 1.594863 | -0.947176 |
| fly25_2d | 0.087137 | 0.047894 | 0.817308 |
| rr25_7d | -0.112824 | -0.053981 | -0.586151 |
| fly25_1d | 0.071836 | 0.046307 | 0.548548 |

### Regime 1
| feature | regime_mean | global_mean | z_diff |
| --- | --- | --- | --- |
| spread_last | 25.692308 | 0.956044 | 10.319058 |
| spread_bps_last | 6.233221 | 0.292986 | 9.995328 |
| spread_mean | 5.759604 | 1.125001 | 4.421400 |
| spread_std | 6.303086 | 1.594863 | 3.431130 |
| call25_iv_7d | 1.224263 | 0.615548 | 2.858195 |
| call25_iv_2d | 1.196169 | 0.591452 | 2.647642 |
| implied_move_1h_pct_1sigma | 0.012436 | 0.005937 | 2.589739 |
| implied_move_24h_pct_1sigma | 0.060922 | 0.029084 | 2.589739 |
| implied_move_4h_pct_1sigma | 0.024871 | 0.011874 | 2.589739 |
| atm_iv_1d | 1.164319 | 0.555841 | 2.589739 |
| call25_iv_1d | 1.193476 | 0.583002 | 2.581282 |
| atm_iv_2d | 1.134038 | 0.566053 | 2.538420 |
| atm_iv_7d | 1.089955 | 0.585052 | 2.515267 |
| put25_iv_7d | 1.263186 | 0.669531 | 2.414798 |
| put25_iv_2d | 1.276797 | 0.636452 | 2.412907 |

### Regime 2
| feature | regime_mean | global_mean | z_diff |
| --- | --- | --- | --- |
| local_timestamp_last | 1725736688306449.750000 | 1673536467280996.750000 | 1.022574 |
| quote_updates | 966389.795483 | 609291.982906 | 0.848403 |
| bid_last | 73129.804266 | 47411.950183 | 0.842308 |
| mid_last | 73130.054266 | 47412.428205 | 0.842302 |
| microprice_last | 73130.057831 | 47412.439101 | 0.842302 |
| ask_last | 73130.304266 | 47412.906227 | 0.842296 |
| spot_opt | 73200.005293 | 47462.779993 | 0.841668 |
| implied_move_24h_usd_1sigma | 1944.012327 | 1288.578429 | 0.811014 |
| implied_move_4h_usd_1sigma | 793.639709 | 526.059941 | 0.811014 |
| implied_move_1h_usd_1sigma | 396.819855 | 263.029970 | 0.811014 |
| imbalance_std | 0.701344 | 0.726517 | -0.457956 |
| spread_bps_last | 0.071043 | 0.292986 | -0.373451 |
| fly25_7d | 0.036156 | 0.057483 | -0.368007 |
| fly25_2d | 0.030263 | 0.047894 | -0.367215 |
| put25_iv_7d | 0.586477 | 0.669531 | -0.337835 |

### Regime 3
| feature | regime_mean | global_mean | z_diff |
| --- | --- | --- | --- |
| call25_iv_1d | 1.011894 | 0.583002 | 1.813497 |
| call25_iv_2d | 1.004190 | 0.591452 | 1.807098 |
| implied_move_4h_pct_1sigma | 0.020882 | 0.011874 | 1.794966 |
| implied_move_1h_pct_1sigma | 0.010441 | 0.005937 | 1.794966 |
| atm_iv_1d | 0.977581 | 0.555841 | 1.794966 |
| implied_move_24h_pct_1sigma | 0.051151 | 0.029084 | 1.794966 |
| call25_iv_7d | 0.996148 | 0.615548 | 1.787089 |
| atm_iv_2d | 0.958836 | 0.566053 | 1.755412 |
| put25_iv_1d | 1.100999 | 0.621303 | 1.733839 |
| put25_iv_2d | 1.082755 | 0.636452 | 1.681731 |
| spread_mean | 2.850619 | 1.125001 | 1.646235 |
| atm_iv_7d | 0.913692 | 0.585052 | 1.637181 |
| spread_std | 3.508405 | 1.594863 | 1.394499 |
| put25_iv_7d | 1.010125 | 0.669531 | 1.385426 |
| term_slope_1d_7d | 0.063889 | -0.029211 | 1.155977 |

### Regime 4
| feature | regime_mean | global_mean | z_diff |
| --- | --- | --- | --- |
| spread_bps_last | 1.534083 | 0.292986 | 2.088331 |
| spread_last | 5.737500 | 0.956044 | 1.994647 |
| spread_mean | 2.755971 | 1.125001 | 1.555941 |
| atm_iv_7d | 0.862102 | 0.585052 | 1.380173 |
| put25_iv_7d | 0.997482 | 0.669531 | 1.334002 |
| spread_std | 3.418802 | 1.594863 | 1.329201 |
| atm_iv_2d | 0.862908 | 0.566053 | 1.326693 |
| implied_move_4h_pct_1sigma | 0.018437 | 0.011874 | 1.307644 |
| implied_move_1h_pct_1sigma | 0.009218 | 0.005937 | 1.307644 |
| implied_move_24h_pct_1sigma | 0.045160 | 0.029084 | 1.307644 |
| atm_iv_1d | 0.863081 | 0.555841 | 1.307644 |
| put25_iv_2d | 0.980464 | 0.636452 | 1.296285 |
| call25_iv_2d | 0.885767 | 0.591452 | 1.288603 |
| put25_iv_1d | 0.974473 | 0.621303 | 1.276519 |
| call25_iv_7d | 0.887245 | 0.615548 | 1.275741 |

### Regime 5
| feature | regime_mean | global_mean | z_diff |
| --- | --- | --- | --- |
| implied_move_4h_usd_1sigma | 241.670129 | 526.059941 | -0.861964 |
| implied_move_1h_usd_1sigma | 120.835065 | 263.029970 | -0.861964 |
| implied_move_24h_usd_1sigma | 591.968503 | 1288.578429 | -0.861964 |
| quote_updates | 315013.786538 | 609291.982906 | -0.699155 |
| spot_opt | 26963.116633 | 47462.779993 | -0.670387 |
| ask_last | 26959.163942 | 47412.906227 | -0.669901 |
| microprice_last | 26958.914975 | 47412.439101 | -0.669892 |
| mid_last | 26958.913942 | 47412.428205 | -0.669892 |
| bid_last | 26958.663942 | 47411.950183 | -0.669884 |
| spread_std | 0.684654 | 1.594863 | -0.663317 |
| atm_iv_7d | 0.474423 | 0.585052 | -0.551117 |
| atm_iv_2d | 0.444033 | 0.566053 | -0.545331 |
| put25_iv_2d | 0.492053 | 0.636452 | -0.544114 |
| atm_iv_1d | 0.428454 | 0.555841 | -0.542171 |
| implied_move_4h_pct_1sigma | 0.009152 | 0.011874 | -0.542171 |

### Regime 6
| feature | regime_mean | global_mean | z_diff |
| --- | --- | --- | --- |
| atm_iv_7d | 0.726113 | 0.585052 | 0.702721 |
| spread_std | 2.551884 | 1.594863 | 0.697432 |
| local_timestamp_last | 1638892783137447.500000 | 1673536467280996.750000 | -0.678651 |
| atm_iv_2d | 0.708281 | 0.566053 | 0.635638 |
| implied_move_1h_pct_1sigma | 0.007474 | 0.005937 | 0.612565 |
| implied_move_24h_pct_1sigma | 0.036615 | 0.029084 | 0.612565 |
| implied_move_4h_pct_1sigma | 0.014948 | 0.011874 | 0.612565 |
| atm_iv_1d | 0.699768 | 0.555841 | 0.612565 |
| put25_iv_7d | 0.818524 | 0.669531 | 0.606055 |
| call25_iv_7d | 0.743799 | 0.615548 | 0.602194 |
| put25_iv_2d | 0.793337 | 0.636452 | 0.591166 |
| call25_iv_2d | 0.722740 | 0.591452 | 0.574821 |
| put25_iv_1d | 0.778935 | 0.621303 | 0.569756 |
| call25_iv_1d | 0.716803 | 0.583002 | 0.565755 |
| imbalance_std | 0.756838 | 0.726517 | 0.551613 |

### Regime 7
| feature | regime_mean | global_mean | z_diff |
| --- | --- | --- | --- |
| bid_last | 107270.445796 | 47411.950183 | 1.960479 |
| mid_last | 107270.695796 | 47412.428205 | 1.960474 |
| microprice_last | 107270.701658 | 47412.439101 | 1.960474 |
| ask_last | 107270.945796 | 47412.906227 | 1.960469 |
| spot_opt | 107354.643404 | 47462.779993 | 1.958605 |
| local_timestamp_last | 1750687645796122.000000 | 1673536467280996.750000 | 1.511350 |
| bid_amt_last | 347388.716814 | 117377.255189 | 1.223526 |
| ask_amt_last | 351717.212389 | 114992.024420 | 1.216721 |
| call25_iv_7d | 0.393351 | 0.615548 | -1.043317 |
| call25_iv_2d | 0.355399 | 0.591452 | -1.033511 |
| call25_iv_1d | 0.342889 | 0.583002 | -1.015277 |
| atm_iv_7d | 0.381792 | 0.585052 | -1.012576 |
| atm_iv_2d | 0.340253 | 0.566053 | -1.009142 |
| implied_move_1h_pct_1sigma | 0.003444 | 0.005937 | -0.993163 |
| implied_move_4h_pct_1sigma | 0.006889 | 0.011874 | -0.993163 |

## Candle type distribution (optional)
Counts by regime:

| row_0 | long | short | skip |
| --- | --- | --- | --- |
| 0 | 89 | 71 | 345 |
| 1 | 2 | 1 | 10 |
| 2 | 132 | 136 | 529 |
| 3 | 24 | 28 | 82 |
| 4 | 54 | 52 | 174 |
| 5 | 148 | 123 | 769 |
| 6 | 189 | 187 | 498 |
| 7 | 38 | 47 | 367 |

Row-normalized distribution by regime:

| row_0 | long | short | skip |
| --- | --- | --- | --- |
| 0 | 0.1762 | 0.1406 | 0.6832 |
| 1 | 0.1538 | 0.0769 | 0.7692 |
| 2 | 0.1656 | 0.1706 | 0.6637 |
| 3 | 0.1791 | 0.2090 | 0.6119 |
| 4 | 0.1929 | 0.1857 | 0.6214 |
| 5 | 0.1423 | 0.1183 | 0.7394 |
| 6 | 0.2162 | 0.2140 | 0.5698 |
| 7 | 0.0841 | 0.1040 | 0.8119 |

## Figures
- D:\GitHub\bitcoin-probabilistic-learning\pipeline\step_regime_gmm_12h_open\results\figures\bic_aic_vs_k.png
- D:\GitHub\bitcoin-probabilistic-learning\pipeline\step_regime_gmm_12h_open\results\figures\probmax_hist.png
- D:\GitHub\bitcoin-probabilistic-learning\pipeline\step_regime_gmm_12h_open\results\figures\entropy_hist.png
- D:\GitHub\bitcoin-probabilistic-learning\pipeline\step_regime_gmm_12h_open\results\figures\regime_timeline.png
- D:\GitHub\bitcoin-probabilistic-learning\pipeline\step_regime_gmm_12h_open\results\figures\regime_transition_heatmap.png
- D:\GitHub\bitcoin-probabilistic-learning\pipeline\step_regime_gmm_12h_open\results\figures\regime_return_boxplot.png
- D:\GitHub\bitcoin-probabilistic-learning\pipeline\step_regime_gmm_12h_open\results\figures\run_length_hist.png

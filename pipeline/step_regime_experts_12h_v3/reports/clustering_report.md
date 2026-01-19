# Clustering Report

## Feature Sets
- Cluster features: log_return_12h, range_pct, body_pct, realized_vol_rolling_20, volume_z_rolling_20, spread_roll_mean, spread_roll_std, imbalance_roll_mean, imbalance_roll_std, iv_slope_atm, rr_slope, fly_slope
- Model features count: 65

## GMM Selection (Full Data)
- K=2: BIC=60415.06, AIC=59367.24, silhouette=0.2781
- K=3: BIC=47042.21, AIC=45467.59, silhouette=0.0425
- K=4: BIC=39602.35, AIC=37500.93, silhouette=0.0075
- K=5: BIC=35292.79, AIC=32664.57, silhouette=0.0090
- K=6: BIC=34836.15, AIC=31681.13, silhouette=0.0047
- K=7: BIC=32071.97, AIC=28390.14, silhouette=0.0001
- K=8: BIC=31858.51, AIC=27649.87, silhouette=0.0051

Chosen K (BIC): 8

## HMM Selection (Full Data)
- K=2: loglik=-33110.18, BIC=66617.60, AIC=66322.35
- K=3: loglik=-28858.26, BIC=58339.64, AIC=57876.52
- K=4: loglik=-27566.40, BIC=55997.38, AIC=55354.80
- K=5: loglik=-26477.82, BIC=54077.25, AIC=53243.63
- K=6: loglik=-25970.21, BIC=53334.66, AIC=52298.42
- K=7: loglik=-25310.49, BIC=52303.41, AIC=51052.98
- K=8: loglik=-25217.11, BIC=52420.42, AIC=50944.22

Chosen K (BIC): 7
# Clustering Report

## Feature Sets
- Cluster features: log_return_12h, range_pct, body_pct, realized_vol_rolling_20, volume_z_rolling_20, spread_bps_rolling_mean, spread_bps_rolling_std, imbalance_rolling_mean, imbalance_rolling_std, iv_slope_atm, rr_change_1d_7d, fly_change_1d_7d
- Model features count: 65

## GMM Selection (Full Data, no CV)
- K=2: BIC=63664.44, AIC=62616.62, silhouette=0.2585
- K=3: BIC=56918.53, AIC=55343.91, silhouette=0.0437
- K=4: BIC=56224.61, AIC=54123.18, silhouette=0.0525
- K=5: BIC=41384.78, AIC=38756.55, silhouette=0.0095
- K=6: BIC=39105.73, AIC=35950.70, silhouette=-0.0029
- K=7: BIC=35974.57, AIC=32292.74, silhouette=-0.0056
- K=8: BIC=40224.96, AIC=36016.33, silhouette=-0.0560

Chosen K (BIC): 7

## HMM Selection (Full Data, no CV)
- K=2: loglik=-34735.24, BIC=69867.72, AIC=69572.48
- K=3: loglik=-30484.41, BIC=61591.94, AIC=61128.82
- K=4: loglik=-29190.28, BIC=59245.15, AIC=58602.57
- K=5: loglik=-28101.47, BIC=57324.56, AIC=56490.94
- K=6: loglik=-27354.03, BIC=56102.29, AIC=55066.05
- K=7: loglik=-26858.50, BIC=55399.43, AIC=54149.00
- K=8: loglik=-25340.23, BIC=52666.67, AIC=51190.46

Chosen K (BIC): 8
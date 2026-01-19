# Clustering report

- Samples used: 4052
- Features used: 59
- Scaler: RobustScaler
- Outlier clipping: True
- Selection method: gmm
- Selected K: 3
- Selected cov_type: diag
- BIC: 560620.142443159
- Silhouette (sampled): 0.09410959890667635

## GMM search
[
  {
    "k": 2,
    "cov_type": "diag",
    "bic": 634109.1925084506,
    "aic": 632614.4415983582,
    "silhouette": 0.32576280388894735
  },
  {
    "k": 3,
    "cov_type": "diag",
    "bic": 560620.142443159,
    "aic": 558374.8625950878,
    "silhouette": 0.09410959890667635
  },
  {
    "k": 4,
    "cov_type": "diag",
    "bic": 561035.4706471376,
    "aic": 558039.6618610875,
    "silhouette": 0.11098068246465351
  }
]

## HMM search
[
  {
    "k": 2,
    "cov_type": "diag",
    "bic": 638765.2320656206,
    "aic": NaN,
    "silhouette": 0.3365872958867875,
    "log_likelihood": -318389.93361189874
  },
  {
    "k": 3,
    "cov_type": "diag",
    "bic": 560519.7823685993,
    "aic": NaN,
    "silhouette": 0.06707928864057218,
    "log_likelihood": -278756.33036266797
  },
  {
    "k": 4,
    "cov_type": "diag",
    "bic": 611400.9134814093,
    "aic": NaN,
    "silhouette": 0.2994960399782661,
    "log_likelihood": -303677.7105524874
  }
]
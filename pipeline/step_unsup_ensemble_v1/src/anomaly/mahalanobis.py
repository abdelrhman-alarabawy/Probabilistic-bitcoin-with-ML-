from __future__ import annotations

import numpy as np
from sklearn.covariance import EmpiricalCovariance, MinCovDet


class MahalanobisAnomaly:
    def __init__(self, robust: bool = True) -> None:
        self.robust = robust
        self.estimator = None

    def fit(self, X_train: np.ndarray) -> "MahalanobisAnomaly":
        if self.robust:
            self.estimator = MinCovDet().fit(X_train)
        else:
            self.estimator = EmpiricalCovariance().fit(X_train)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.estimator is None:
            raise ValueError("MahalanobisAnomaly not fitted.")
        return self.estimator.mahalanobis(X).astype(float)

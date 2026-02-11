from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest


class IsolationForestAnomaly:
    def __init__(self, contamination: float = 0.01, seed: int = 42) -> None:
        self.model = IsolationForest(
            contamination=contamination,
            random_state=seed,
            n_estimators=300,
        )

    def fit(self, X_train: np.ndarray) -> "IsolationForestAnomaly":
        self.model.fit(X_train)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        # Higher score should mean more anomalous.
        normality = self.model.score_samples(X)
        return (-normality).astype(float)

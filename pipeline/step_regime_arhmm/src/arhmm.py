from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


class NonPDMatrixError(RuntimeError):
    pass


def _logsumexp(a: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    a_max = np.max(a, axis=axis, keepdims=True)
    out = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    if axis is None:
        return out.reshape(())
    return np.squeeze(out, axis=axis)


def _normalize_probs(p: np.ndarray, axis: int = -1, min_prob: float = 1e-12) -> np.ndarray:
    p = np.clip(p, min_prob, None)
    p = p / p.sum(axis=axis, keepdims=True)
    return p


def _log_gaussian_ar(
    x_curr: np.ndarray,
    x_lag: np.ndarray,
    coef: np.ndarray,
    intercept: np.ndarray,
    cov: np.ndarray,
) -> np.ndarray:
    mean = x_lag @ coef.T + intercept
    diff = x_curr - mean
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as exc:
        raise NonPDMatrixError("covariance not PD") from exc

    solve = np.linalg.solve(chol, diff.T)
    maha = np.sum(solve**2, axis=0)
    logdet = 2.0 * np.sum(np.log(np.diag(chol)))
    d = x_curr.shape[1]
    return -0.5 * (d * np.log(2.0 * np.pi) + logdet + maha)


@dataclass
class ARHMMFitResult:
    loglik: float
    loglik_trace: List[float]
    converged: bool
    n_iter: int


class ARHMM:
    def __init__(
        self,
        n_states: int,
        n_features: int,
        eps: float = 1e-6,
        max_iter: int = 200,
        tol: float = 1e-4,
        seed: int = 0,
        use_ledoitwolf: bool = False,
    ) -> None:
        self.n_states = int(n_states)
        self.n_features = int(n_features)
        self.eps = float(eps)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.seed = int(seed)
        self.use_ledoitwolf = bool(use_ledoitwolf)

        rng = np.random.RandomState(self.seed)
        self.pi = _normalize_probs(rng.rand(self.n_states))
        self.A = _normalize_probs(rng.rand(self.n_states, self.n_states), axis=1)
        self.coef = rng.randn(self.n_states, self.n_features, self.n_features) * 0.05
        self.intercept = rng.randn(self.n_states, self.n_features) * 0.05
        self.cov = np.array([np.eye(self.n_features) for _ in range(self.n_states)])

    def _compute_log_emissions(self, x_curr: np.ndarray, x_lag: np.ndarray) -> np.ndarray:
        t = x_curr.shape[0]
        log_emissions = np.zeros((t, self.n_states))
        for k in range(self.n_states):
            log_emissions[:, k] = _log_gaussian_ar(
                x_curr=x_curr,
                x_lag=x_lag,
                coef=self.coef[k],
                intercept=self.intercept[k],
                cov=self.cov[k],
            )
        return log_emissions

    def _forward_backward(
        self, log_emissions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        t, k = log_emissions.shape
        log_pi = np.log(_normalize_probs(self.pi))
        log_A = np.log(_normalize_probs(self.A, axis=1))

        log_alpha = np.zeros((t, k))
        log_alpha[0] = log_pi + log_emissions[0]
        for i in range(1, t):
            log_alpha[i] = log_emissions[i] + _logsumexp(log_alpha[i - 1][:, None] + log_A, axis=0)

        loglik = float(_logsumexp(log_alpha[-1], axis=0))

        log_beta = np.zeros((t, k))
        for i in range(t - 2, -1, -1):
            log_beta[i] = _logsumexp(log_A + log_emissions[i + 1] + log_beta[i + 1], axis=1)

        log_gamma = log_alpha + log_beta - loglik
        gamma = np.exp(log_gamma)

        xi_sum = np.zeros((k, k))
        for i in range(t - 1):
            log_xi = (
                log_alpha[i][:, None]
                + log_A
                + log_emissions[i + 1][None, :]
                + log_beta[i + 1][None, :]
                - loglik
            )
            xi_sum += np.exp(log_xi)

        return gamma, xi_sum, loglik

    def _m_step(
        self, x_curr: np.ndarray, x_lag: np.ndarray, gamma: np.ndarray, xi_sum: np.ndarray
    ) -> None:
        t, d = x_curr.shape
        k = self.n_states

        self.pi = _normalize_probs(gamma[0])

        denom = gamma[:-1].sum(axis=0)
        A = np.zeros((k, k))
        for i in range(k):
            if denom[i] <= 0:
                A[i] = 1.0 / k
            else:
                A[i] = xi_sum[i] / denom[i]
        self.A = _normalize_probs(A, axis=1)

        x_aug = np.hstack([x_lag, np.ones((t, 1))])
        ridge = self.eps * np.eye(d + 1)

        for state in range(k):
            w = gamma[:, state]
            n_eff = float(w.sum())
            if n_eff <= 0:
                self.coef[state] = 0.0
                self.intercept[state] = 0.0
                self.cov[state] = np.eye(d) * (1.0 + self.eps)
                continue

            xtw = x_aug.T * w
            xtwx = xtw @ x_aug
            try:
                beta = np.linalg.solve(xtwx + ridge, xtw @ x_curr)
            except np.linalg.LinAlgError:
                beta = np.linalg.lstsq(xtwx + ridge, xtw @ x_curr, rcond=None)[0]

            self.coef[state] = beta[:d].T
            self.intercept[state] = beta[d]

            resid = x_curr - x_aug @ beta
            cov = (resid.T * w) @ resid / n_eff
            cov = 0.5 * (cov + cov.T)

            if self.use_ledoitwolf:
                try:
                    from sklearn.covariance import LedoitWolf

                    cov = LedoitWolf().fit(resid).covariance_
                except Exception:
                    pass

            cov = cov + self.eps * np.eye(d)
            self.cov[state] = cov

    def fit(self, x: np.ndarray) -> ARHMMFitResult:
        if x.ndim != 2:
            raise ValueError("x must be 2D array")
        if x.shape[0] < 2:
            raise ValueError("Need at least 2 time steps for AR(1)")

        x_lag = x[:-1]
        x_curr = x[1:]

        loglik_trace: List[float] = []
        prev_loglik = None
        converged = False

        for n_iter in range(1, self.max_iter + 1):
            log_emissions = self._compute_log_emissions(x_curr, x_lag)
            gamma, xi_sum, loglik = self._forward_backward(log_emissions)
            self._m_step(x_curr, x_lag, gamma, xi_sum)
            loglik_trace.append(loglik)

            if prev_loglik is not None:
                if abs(loglik - prev_loglik) < self.tol:
                    converged = True
                    break
            prev_loglik = loglik

        return ARHMMFitResult(
            loglik=float(loglik_trace[-1]),
            loglik_trace=loglik_trace,
            converged=converged,
            n_iter=n_iter,
        )

    def loglikelihood(self, x: np.ndarray) -> float:
        if x.shape[0] < 2:
            raise ValueError("Need at least 2 time steps for AR(1)")
        x_lag = x[:-1]
        x_curr = x[1:]
        log_emissions = self._compute_log_emissions(x_curr, x_lag)
        _, _, loglik = self._forward_backward(log_emissions)
        return float(loglik)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] < 2:
            raise ValueError("Need at least 2 time steps for AR(1)")
        x_lag = x[:-1]
        x_curr = x[1:]
        log_emissions = self._compute_log_emissions(x_curr, x_lag)
        gamma, _, _ = self._forward_backward(log_emissions)
        return gamma

    def viterbi(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] < 2:
            raise ValueError("Need at least 2 time steps for AR(1)")
        x_lag = x[:-1]
        x_curr = x[1:]
        log_emissions = self._compute_log_emissions(x_curr, x_lag)

        t, k = log_emissions.shape
        log_pi = np.log(_normalize_probs(self.pi))
        log_A = np.log(_normalize_probs(self.A, axis=1))

        delta = np.zeros((t, k))
        psi = np.zeros((t, k), dtype=int)
        delta[0] = log_pi + log_emissions[0]

        for i in range(1, t):
            scores = delta[i - 1][:, None] + log_A
            psi[i] = np.argmax(scores, axis=0)
            delta[i] = log_emissions[i] + np.max(scores, axis=0)

        path = np.zeros(t, dtype=int)
        path[-1] = int(np.argmax(delta[-1]))
        for i in range(t - 2, -1, -1):
            path[i] = psi[i + 1, path[i + 1]]

        return path

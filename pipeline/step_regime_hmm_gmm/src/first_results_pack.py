from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import davies_bouldin_score, silhouette_score


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _fmt(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "NA"
    try:
        if not np.isfinite(value):
            return "NA"
    except Exception:
        return "NA"
    return f"{value:.{digits}f}"


def _logsumexp(arr: np.ndarray, axis: int = 1) -> np.ndarray:
    maxv = np.max(arr, axis=axis, keepdims=True)
    return maxv + np.log(np.sum(np.exp(arr - maxv), axis=axis, keepdims=True))


def _log_gaussian(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    d = mean.shape[0]
    cov = cov + np.eye(d) * 1.0e-9
    try:
        chol = np.linalg.cholesky(cov)
        diff = (X - mean).T
        sol = np.linalg.solve(chol, diff)
        quad = np.sum(sol * sol, axis=0)
        logdet = 2.0 * np.sum(np.log(np.diag(chol)))
    except Exception:
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            logdet = np.log(np.abs(np.linalg.det(cov)) + 1.0e-12)
        diff = X - mean
        quad = np.sum(diff * np.dot(diff, np.linalg.inv(cov)), axis=1)
    return -0.5 * (d * np.log(2 * np.pi) + logdet + quad)


def _get_covar(model, state: int, mix: int) -> np.ndarray:
    covars = model.covars_
    if covars.ndim == 4:
        return covars[state, mix]
    if covars.ndim == 3:
        if covars.shape[0] == model.n_components:
            return covars[state]
        if covars.shape[0] == model.n_mix:
            return covars[mix]
        return covars[0]
    if covars.ndim == 2:
        return covars
    raise ValueError("Unsupported covariance shape")


def _component_logpdfs(model, X: np.ndarray, state: int) -> np.ndarray:
    n_mix = model.n_mix if hasattr(model, "n_mix") else 1
    logpdfs = np.zeros((X.shape[0], n_mix))
    for m in range(n_mix):
        mean = model.means_[state, m] if n_mix > 1 else model.means_[state]
        cov = _get_covar(model, state, m)
        logpdfs[:, m] = _log_gaussian(X, mean, cov)
    return logpdfs


def _responsibilities(model, X: np.ndarray, state: int) -> np.ndarray:
    n_mix = model.n_mix if hasattr(model, "n_mix") else 1
    if n_mix == 1:
        return np.ones((X.shape[0], 1))
    logpdfs = _component_logpdfs(model, X, state)
    logw = np.log(np.clip(model.weights_[state], 1.0e-12, 1.0))
    log_joint = logpdfs + logw
    log_norm = _logsumexp(log_joint, axis=1)
    resp = np.exp(log_joint - log_norm)
    return resp


def _state_entropy(posteriors: np.ndarray) -> np.ndarray:
    eps = 1e-12
    probs = np.clip(posteriors, eps, 1.0)
    return -np.sum(probs * np.log(probs), axis=1)


def _count_switches(states: np.ndarray) -> int:
    if states.size <= 1:
        return 0
    return int(np.sum(states[1:] != states[:-1]))


def _expected_durations(transmat: np.ndarray) -> np.ndarray:
    diag = np.clip(np.diag(transmat), 1e-9, 1 - 1e-9)
    return 1.0 / (1.0 - diag)


def _safe_cluster_metrics(
    X: np.ndarray, labels: np.ndarray, max_samples: int, rng: np.random.Generator
) -> Tuple[Optional[float], Optional[float]]:
    if X.shape[0] < 3 or len(np.unique(labels)) < 2:
        return None, None

    if X.shape[0] > max_samples:
        idx = rng.choice(X.shape[0], size=max_samples, replace=False)
        X_eval = X[idx]
        labels_eval = labels[idx]
    else:
        X_eval = X
        labels_eval = labels

    try:
        sil = float(silhouette_score(X_eval, labels_eval))
    except Exception:
        sil = None
    try:
        db = float(davies_bouldin_score(X_eval, labels_eval))
    except Exception:
        db = None
    return sil, db


def _covariance_health(model) -> Tuple[bool, bool]:
    n_states = model.n_components
    n_mix = model.n_mix if hasattr(model, "n_mix") else 1
    any_non_pd = False
    any_near_zero = False
    for state in range(n_states):
        for mix in range(n_mix):
            try:
                cov = _get_covar(model, state, mix)
                if not np.isfinite(cov).all():
                    any_non_pd = True
                diag = np.diag(cov)
                if np.min(diag) < 1.0e-8:
                    any_near_zero = True
                try:
                    np.linalg.cholesky(cov)
                except Exception:
                    any_non_pd = True
            except Exception:
                any_non_pd = True
    return any_near_zero, any_non_pd


def _select_gmm_rows(
    gmm_runs_df,
    gmm_summary_df,
    k: int,
    cov_type: str,
) -> Tuple[Optional[dict], Optional[dict]]:
    if gmm_runs_df is None or gmm_runs_df.empty:
        return None, None

    subset = gmm_runs_df[(gmm_runs_df["k"] == k) & (gmm_runs_df["cov_type"] == cov_type)]
    if subset.empty:
        subset = gmm_runs_df[gmm_runs_df["k"] == k]
    if subset.empty:
        subset = gmm_runs_df
    best_row = subset.sort_values("test_loglik_avg", ascending=False).iloc[0].to_dict()

    summary_row = None
    if gmm_summary_df is not None and not gmm_summary_df.empty:
        summary_subset = gmm_summary_df[
            (gmm_summary_df["k"] == k) & (gmm_summary_df["cov_type"] == cov_type)
        ]
        if summary_subset.empty:
            summary_subset = gmm_summary_df[gmm_summary_df["k"] == k]
        if not summary_subset.empty:
            summary_row = summary_subset.iloc[0].to_dict()
    return best_row, summary_row


def _json_dump(path: Path, payload: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def print_first_results_pack(run_context: Dict[str, object]) -> Dict[str, object]:
    fold_id = str(run_context.get("fold_id", ""))
    feature_set = str(run_context.get("feature_set", ""))
    period = run_context.get("period", {}) or {}
    config = run_context.get("config", {}) or {}
    data_info = run_context.get("data", {}) or {}
    params = run_context.get("params", {}) or {}

    k = int(config.get("k", 0))
    n_mix = int(config.get("n_mix", 1))
    cov_type = str(config.get("cov_type", ""))
    seed = int(config.get("seed", 0))

    n_train = int(data_info.get("n_train", 0))
    n_test = int(data_info.get("n_test", 0))
    n_features = int(data_info.get("n_features", 0))

    print(
        "FIRST_RESULTS_PACK | fold=%s train=%s..%s test=%s..%s | K=%d M=%d cov=%s seed=%d | n_train=%d n_test=%d n_features=%d"
        % (
            fold_id,
            period.get("train_start", ""),
            period.get("train_end", ""),
            period.get("test_start", ""),
            period.get("test_end", ""),
            k,
            n_mix,
            cov_type,
            seed,
            n_train,
            n_test,
            n_features,
        )
    )

    gmm_best, gmm_summary = _select_gmm_rows(
        run_context.get("gmm_runs_df"),
        run_context.get("gmm_summary_df"),
        k,
        cov_type,
    )

    gmm_ari = "NA"
    gmm_ari_mean = np.nan
    gmm_ari_std = np.nan
    if gmm_summary and int(gmm_summary.get("n_runs", 1)) > 1:
        gmm_ari_mean = _as_float(gmm_summary.get("ari_mean", np.nan))
        gmm_ari_std = _as_float(gmm_summary.get("ari_std", np.nan))
        gmm_ari = "%s±%s" % (_fmt(gmm_ari_mean), _fmt(gmm_ari_std))

    gmm_summary_line = {
        "gmm_train_avg_loglik": _as_float(gmm_best.get("train_loglik_avg")) if gmm_best else np.nan,
        "gmm_test_avg_loglik": _as_float(gmm_best.get("test_loglik_avg")) if gmm_best else np.nan,
        "gmm_AIC": _as_float(gmm_best.get("train_aic")) if gmm_best else np.nan,
        "gmm_BIC": _as_float(gmm_best.get("train_bic")) if gmm_best else np.nan,
        "gmm_resp_entropy_train_mean": _as_float(gmm_best.get("resp_entropy_mean")) if gmm_best else np.nan,
        "gmm_stability_ari": gmm_ari,
    }
    print("GMM_BASELINE %s" % json.dumps(gmm_summary_line, separators=(",", ":")))

    hmm = run_context.get("hmm", {}) or {}
    model = hmm.get("model")
    post_train = hmm.get("post_train")
    post_test = hmm.get("post_test")
    viterbi_test = hmm.get("viterbi_test")
    hmm_run = hmm.get("run")

    transmat = model.transmat_ if model is not None else np.zeros((k, k))
    mean_diag = float(np.mean(np.diag(transmat))) if transmat.size else np.nan
    expected_durations = _expected_durations(transmat) if transmat.size else np.array([])

    post_entropy = np.array([])
    if isinstance(post_test, np.ndarray) and post_test.size:
        post_entropy = _state_entropy(post_test)
    if post_entropy.size:
        ent_mean = float(np.mean(post_entropy))
        p10, p50, p90 = np.percentile(post_entropy, [10, 50, 90])
    else:
        ent_mean = np.nan
        p10 = np.nan
        p50 = np.nan
        p90 = np.nan

    occ_train = {}
    occ_test = {}
    n_eff_train = {}
    n_eff_test = {}
    if isinstance(post_train, np.ndarray) and post_train.size:
        occ_train = {int(i): float(np.mean(post_train[:, i])) for i in range(post_train.shape[1])}
        n_eff_train = {int(i): float(np.sum(post_train[:, i])) for i in range(post_train.shape[1])}
    if isinstance(post_test, np.ndarray) and post_test.size:
        occ_test = {int(i): float(np.mean(post_test[:, i])) for i in range(post_test.shape[1])}
        n_eff_test = {int(i): float(np.sum(post_test[:, i])) for i in range(post_test.shape[1])}

    switches = _count_switches(viterbi_test) if isinstance(viterbi_test, np.ndarray) else 0

    hmm_summary_line = {
        "hmm_train_avg_loglik": _as_float(getattr(hmm_run, "train_loglik_avg", np.nan)),
        "hmm_test_avg_loglik": _as_float(getattr(hmm_run, "test_loglik_avg", np.nan)),
        "hmm_AIC": _as_float(getattr(hmm_run, "aic", np.nan)),
        "hmm_BIC": _as_float(getattr(hmm_run, "bic", np.nan)),
        "posterior_entropy_mean": ent_mean,
        "posterior_entropy_p10": _as_float(p10),
        "posterior_entropy_p50": _as_float(p50),
        "posterior_entropy_p90": _as_float(p90),
        "mean_diag_A": mean_diag,
        "n_switches_viterbi": int(switches),
        "occupancy_soft_train": occ_train,
        "occupancy_soft_test": occ_test,
        "N_eff_train": n_eff_train,
        "N_eff_test": n_eff_test,
        "expected_duration": expected_durations.tolist() if expected_durations.size else [],
    }
    print("HMM_GMM %s" % json.dumps(hmm_summary_line, separators=(",", ":")))

    if transmat.size:
        print("TRANSITION_MATRIX")
        for row in transmat:
            print("  [" + " ".join(_fmt(float(v), digits=3) for v in row) + "]")
        tops = []
        for i in range(transmat.shape[0]):
            j = int(np.argmax(transmat[i]))
            tops.append("s%d->s%d:%s" % (i, j, _fmt(float(transmat[i, j]), digits=3)))
        print("TOP_TRANSITIONS " + " ".join(tops))

    warnings: List[str] = []
    if isinstance(post_train, np.ndarray) and post_train.size:
        t_train = post_train.shape[0]
        threshold = max(50, int(0.05 * t_train))
        for state in range(post_train.shape[1]):
            n_eff = float(np.sum(post_train[:, state]))
            occ = float(np.mean(post_train[:, state]))
            if n_eff < threshold or occ < 0.03:
                warnings.append("Ghost state %d: N_eff=%.2f occ=%.3f" % (state, n_eff, occ))

    if transmat.size:
        for i, val in enumerate(np.diag(transmat)):
            if val > 0.995:
                warnings.append("Sticky state %d: A_kk=%.4f" % (i, float(val)))

    if model is not None:
        any_near_zero, any_non_pd = _covariance_health(model)
        if any_near_zero:
            warnings.append("Collapse risk: near-zero variance detected in emissions.")
        if any_non_pd:
            warnings.append("Collapse risk: non-PD covariance detected in emissions.")

    if warnings:
        print("WARNINGS")
        for warning in warnings:
            print("- %s" % warning)
    else:
        print("WARNINGS OK")

    per_state_rows: List[Dict[str, object]] = []
    state_quality_rows: List[Dict[str, object]] = []
    tau = float(params.get("tau", 0.4))
    max_samples = int(params.get("max_samples_metrics", 2000))
    rng = np.random.default_rng(int(params.get("seed", 0)))

    X_train = run_context.get("arrays", {}).get("X_train") if run_context.get("arrays") else None
    if model is not None and isinstance(post_train, np.ndarray) and isinstance(X_train, np.ndarray):
        print("STATE_INTERNAL_GMM")
        for state in range(model.n_components):
            try:
                gamma_k = post_train[:, state]
                n_eff = float(np.sum(gamma_k))
                occ = float(np.mean(gamma_k)) if gamma_k.size else np.nan
                resp = _responsibilities(model, X_train, state)
                masses = (gamma_k[:, None] * resp).sum(axis=0) if resp.size else np.array([])
                if n_eff > 0 and masses.size:
                    masses_norm = (masses / n_eff).tolist()
                else:
                    masses_norm = [float("nan")] * resp.shape[1] if resp.size else []
                eff_comps = int(np.sum(np.array(masses_norm) > 0.05)) if masses_norm else 0
                if resp.size and n_eff > 0:
                    eps = 1.0e-12
                    ent = -np.sum(np.clip(resp, eps, 1.0) * np.log(resp + eps), axis=1)
                    h_within = float(np.sum(gamma_k * ent) / n_eff)
                else:
                    h_within = np.nan
            except Exception:
                n_eff = np.nan
                occ = np.nan
                masses_norm = []
                eff_comps = 0
                h_within = np.nan

            per_state_rows.append(
                {
                    "state": int(state),
                    "N_eff": n_eff,
                    "occupancy": occ,
                    "component_masses": masses_norm,
                    "effective_components": eff_comps,
                    "within_state_entropy": h_within,
                }
            )
            masses_str = "[" + ",".join(_fmt(float(x), digits=3) for x in masses_norm) + "]"
            print(
                "state %d: N_eff=%s occ=%s masses=%s eff_comps=%d H_within=%s"
                % (state, _fmt(n_eff), _fmt(occ), masses_str, eff_comps, _fmt(h_within))
            )

            try:
                idx = np.where(post_train[:, state] >= tau)[0]
                if idx.size >= 30:
                    resp_sel = _responsibilities(model, X_train[idx], state)
                    labels = np.argmax(resp_sel, axis=1) if resp_sel.size else np.array([])
                    sil, db = _safe_cluster_metrics(X_train[idx], labels, max_samples, rng)
                    sil_val = np.nan if sil is None else sil
                    db_val = np.nan if db is None else db
                else:
                    sil_val = np.nan
                    db_val = np.nan
            except Exception:
                idx = np.array([], dtype=int)
                sil_val = np.nan
                db_val = np.nan
            state_quality_rows.append(
                {
                    "state": int(state),
                    "n_selected": int(idx.size),
                    "silhouette": sil_val,
                    "davies_bouldin": db_val,
                }
            )

        print("STATE_QUALITY tau=%s" % _fmt(tau, digits=2))
        for row in state_quality_rows:
            print(
                "state %d: n_selected=%d silhouette=%s db=%s"
                % (
                    row["state"],
                    row["n_selected"],
                    _fmt(row["silhouette"]),
                    _fmt(row["davies_bouldin"]),
                )
            )

    summary_record: Dict[str, object] = {
        "fold_id": fold_id,
        "feature_set": feature_set,
        "train_start": period.get("train_start"),
        "train_end": period.get("train_end"),
        "test_start": period.get("test_start"),
        "test_end": period.get("test_end"),
        "k": k,
        "n_mix": n_mix,
        "cov_type": cov_type,
        "seed": seed,
        "n_train": n_train,
        "n_test": n_test,
        "n_features": n_features,
        "gmm_summary": gmm_summary_line,
        "hmm_summary": hmm_summary_line,
        "transition_matrix": transmat.tolist() if transmat.size else [],
        "top_transitions": (
            {int(i): int(np.argmax(transmat[i])) for i in range(transmat.shape[0])}
            if transmat.size
            else {}
        ),
        "warnings": warnings,
        "per_state_internal": per_state_rows,
        "state_quality": state_quality_rows,
    }

    output_cfg = run_context.get("output", {}) or {}
    summary_path = Path(output_cfg.get("summary_path", ""))
    if str(summary_path):
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        if summary_path.exists():
            try:
                existing = json.loads(summary_path.read_text(encoding="utf-8"))
                if not isinstance(existing, list):
                    existing = []
            except Exception:
                existing = []
        else:
            existing = []
        existing.append(summary_record)
        _json_dump(summary_path, existing)
        print("FIRST_PACK_SUMMARY_SAVED %s" % str(summary_path))

    return summary_record

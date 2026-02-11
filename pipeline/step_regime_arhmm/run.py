from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.arhmm import ARHMM, NonPDMatrixError
from src.eval import aic_bic, entropy_stats, label_composition, num_params_arhmm, tradeable_states

DEFAULT_DATA_PATH = "data_12h_indicators.csv"
RESULTS_DIR = os.path.join("pipeline", "step_regime_arhmm", "results")
SUMMARY_PATH = os.path.join(RESULTS_DIR, "arhmm_summary.jsonl")

TIMESTAMP_COL = "timestamp"
TIME_START = "2020-01-01"
TIME_END = "2025-12-31"

EXCLUDE_COLUMNS = {
    "timestamp",
    "local_timestamp_last",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "spot_opt",
    "bid_last",
    "ask_last",
    "mid_last",
    "microprice_last",
}

FOLD_SPECS = [
    ("2020_2021_test_2022", [2020, 2021], [2022]),
    ("2020_2022_test_2023", [2020, 2021, 2022], [2023]),
    ("2020_2023_test_2024", [2020, 2021, 2022, 2023], [2024]),
    ("2020_2024_test_2025", [2020, 2021, 2022, 2023, 2024], [2025]),
]

EPS_LIST = [1e-6, 1e-5, 1e-4]
K_LIST = [2, 3, 4]
SEED_LIST = [89250, 773956, 654571]
SUBSET_SIZES = [8, 12, 16, 24, 32]
CORR_THRESHOLD = 0.90


try:
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:
    StandardScaler = None
    SKLEARN_AVAILABLE = False


@dataclass
class Fold:
    name: str
    train_years: List[int]
    test_years: List[int]


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TIMESTAMP_COL not in df.columns:
        raise ValueError(f"Missing timestamp column '{TIMESTAMP_COL}'")
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True, errors="coerce")
    if df[TIMESTAMP_COL].dt.tz is not None:
        df[TIMESTAMP_COL] = df[TIMESTAMP_COL].dt.tz_convert(None)
    df = df.sort_values(TIMESTAMP_COL)
    df = df[(df[TIMESTAMP_COL] >= pd.to_datetime(TIME_START)) & (df[TIMESTAMP_COL] <= pd.to_datetime(TIME_END))]
    return df.reset_index(drop=True)


def detect_label_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ["candle_type", "label", "labels", "signal", "position"]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def indicator_columns(df: pd.DataFrame, label_col: Optional[str]) -> List[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = set(EXCLUDE_COLUMNS)
    if label_col:
        exclude.add(label_col)
    cols = [c for c in numeric_cols if c not in exclude]
    return cols


def load_recommended_features(paths: Iterable[str]) -> Optional[List[str]]:
    for path in paths:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        if "feature" in df.columns:
            feats = df["feature"].astype(str).tolist()
        elif "features" in df.columns:
            feats = df["features"].astype(str).tolist()
        else:
            feats = df.iloc[:, 0].astype(str).tolist()
        feats = [f for f in feats if f]
        if feats:
            return feats
    return None


def variance_fallback_features(df: pd.DataFrame, candidates: List[str], top_n: int = 20) -> List[str]:
    data = df[candidates].dropna().values
    if data.size == 0:
        return candidates[:top_n]

    if SKLEARN_AVAILABLE:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
    else:
        mean = data.mean(axis=0)
        std = data.std(axis=0, ddof=0)
        std[std == 0] = 1.0
        data = (data - mean) / std

    var = data.var(axis=0, ddof=0)
    order = np.argsort(-var)
    sorted_cols = [candidates[i] for i in order]

    if np.allclose(var, var[0], rtol=1e-5, atol=1e-8):
        sorted_cols = sorted(candidates)

    return sorted_cols[:top_n]


def prune_correlated_features(
    x_train: np.ndarray, feature_names: List[str], corr_threshold: float = CORR_THRESHOLD
) -> List[str]:
    if x_train.size == 0 or len(feature_names) <= 1:
        return list(feature_names)

    df = pd.DataFrame(x_train, columns=feature_names)
    corr = df.corr().abs().fillna(0.0).to_numpy()
    variances = np.nanvar(x_train, axis=0)
    nan_counts = np.isnan(x_train).sum(axis=0)

    keep = np.ones(len(feature_names), dtype=bool)
    for i in range(len(feature_names)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(feature_names)):
            if not keep[j]:
                continue
            if corr[i, j] < corr_threshold:
                continue

            var_i = variances[i]
            var_j = variances[j]
            nan_i = nan_counts[i]
            nan_j = nan_counts[j]

            if np.isnan(var_i):
                var_i = -np.inf
            if np.isnan(var_j):
                var_j = -np.inf

            if var_i > var_j:
                keep[j] = False
            elif var_j > var_i:
                keep[i] = False
                break
            else:
                if nan_i < nan_j:
                    keep[j] = False
                elif nan_j < nan_i:
                    keep[i] = False
                    break
                else:
                    keep[j] = False

    return [name for name, flag in zip(feature_names, keep) if flag]


def build_feature_sets(recommended: List[str], prefix: str = "rec") -> OrderedDict:
    feature_sets: OrderedDict[str, List[str]] = OrderedDict()
    if not recommended:
        return feature_sets

    feature_sets[f"{prefix}_full"] = list(recommended)

    for size in SUBSET_SIZES:
        cap = min(size, len(recommended))
        name = f"{prefix}_top{cap}"
        subset = list(recommended[:cap])
        feature_sets.setdefault(name, subset)

    rng = np.random.RandomState(42)
    seen = {tuple(v) for v in feature_sets.values()}
    rand_prefix = "rand_pruned" if "pruned" in prefix else "rand"
    for size in SUBSET_SIZES:
        cap = min(size, len(recommended))
        if cap == 0:
            continue
        target = 3
        attempts = 0
        created = 0
        while created < target and attempts < 50:
            attempts += 1
            idx = rng.choice(len(recommended), size=cap, replace=False)
            idx.sort()
            subset = [recommended[i] for i in idx]
            key = tuple(subset)
            if key in seen:
                continue
            seen.add(key)
            created += 1
            feature_sets[f"{rand_prefix}_s{cap}_r{created}"] = subset

    return feature_sets


def build_random_feature_sets(
    pool: List[str],
    sizes: List[int],
    max_trials: int,
    seed: int = 42,
    allow_repeat_subsets: bool = False,
) -> List[Tuple[str, List[str]]]:
    if not pool:
        return []
    valid_sizes = [s for s in sizes if s <= len(pool)]
    if not valid_sizes:
        valid_sizes = [len(pool)]

    rng = np.random.RandomState(seed)
    seen = set()
    out: List[Tuple[str, List[str]]] = []
    attempts = 0
    while len(out) < max_trials and attempts < max_trials * 10:
        size = valid_sizes[len(out) % len(valid_sizes)]
        idx = rng.choice(len(pool), size=size, replace=False)
        idx.sort()
        subset = [pool[i] for i in idx]
        key = tuple(subset)
        attempts += 1
        if not allow_repeat_subsets:
            if key in seen:
                continue
            seen.add(key)
        out.append((f"search_s{size}_t{len(out)}", subset))
    return out


def build_folds(df: pd.DataFrame) -> List[Fold]:
    years = sorted(df[TIMESTAMP_COL].dt.year.unique().tolist())
    folds: List[Fold] = []
    for name, train_years, test_years in FOLD_SPECS:
        train_avail = [y for y in train_years if y in years]
        test_avail = [y for y in test_years if y in years]
        if not train_avail or not test_avail:
            continue
        fold_name = f"{train_avail[0]}_{train_avail[-1]}_test_{test_avail[0]}"
        folds.append(Fold(name=fold_name, train_years=train_avail, test_years=test_avail))
    return folds


def split_fold(df: pd.DataFrame, fold: Fold) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df[df[TIMESTAMP_COL].dt.year.isin(fold.train_years)].copy()
    test_df = df[df[TIMESTAMP_COL].dt.year.isin(fold.test_years)].copy()
    return train_df, test_df


def fit_scaler(x_train: np.ndarray):
    if SKLEARN_AVAILABLE:
        scaler = StandardScaler()
        scaler.fit(x_train)
        return scaler
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0, ddof=0)
    std[std == 0] = 1.0
    return (mean, std)


def transform_scaler(scaler, x: np.ndarray) -> np.ndarray:
    if SKLEARN_AVAILABLE:
        return scaler.transform(x)
    mean, std = scaler
    return (x - mean) / std


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def format_pct(x: float) -> str:
    return f"{x:.2f}"


def run_one_config(
    fold: Fold,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
    label_col: Optional[str],
    k: int,
    eps: float,
    seed: int,
    max_iter: int,
    tol: float,
    use_ledoitwolf: bool,
    eps_list: List[float],
    corr_threshold: float,
    n_pruned_pool: int,
) -> Tuple[Optional[Dict], Optional[str]]:
    train_df = train_df.dropna(subset=features)
    test_df = test_df.dropna(subset=features)

    if train_df.empty or test_df.empty:
        return None, "empty_train_or_test"

    x_train = train_df[features].values.astype(float)
    x_test = test_df[features].values.astype(float)

    if x_train.shape[0] < 2 or x_test.shape[0] < 2:
        return None, "insufficient_rows"

    scaler = fit_scaler(x_train)
    x_train = transform_scaler(scaler, x_train)
    x_test = transform_scaler(scaler, x_test)

    labels_train = train_df[label_col].values if label_col else None
    labels_test = test_df[label_col].values if label_col else None

    eps_idx = eps_list.index(eps)
    eps_used = eps
    eps_bumped = None
    model = None
    fit_result = None
    last_err = None

    for j in range(eps_idx, len(eps_list)):
        eps_try = eps_list[j]
        model = ARHMM(
            n_states=k,
            n_features=x_train.shape[1],
            eps=eps_try,
            max_iter=max_iter,
            tol=tol,
            seed=seed,
            use_ledoitwolf=use_ledoitwolf,
        )
        try:
            fit_result = model.fit(x_train)
            eps_used = eps_try
            if eps_try != eps:
                eps_bumped = (eps, eps_try)
            break
        except NonPDMatrixError as exc:
            last_err = str(exc)
            continue
        except Exception as exc:
            return None, f"fit_failed: {exc}"

    if fit_result is None or model is None:
        reason = last_err or "fit_failed"
        return None, reason

    train_loglik = model.loglikelihood(x_train)
    test_loglik = model.loglikelihood(x_test)

    n_params = num_params_arhmm(k, x_train.shape[1])
    aic, bic = aic_bic(train_loglik, n_params, x_train.shape[0] - 1)

    gamma_train = model.predict_proba(x_train)
    gamma_test = model.predict_proba(x_test)

    vpath_test = model.viterbi(x_test)
    switches = int(np.sum(vpath_test[1:] != vpath_test[:-1])) if len(vpath_test) > 1 else 0

    occ_train = gamma_train.mean(axis=0)
    occ_test = gamma_test.mean(axis=0)

    n_eff_train = gamma_train.sum(axis=0)
    n_eff_test = gamma_test.sum(axis=0)

    ent = entropy_stats(gamma_test)

    labels_train_aligned = labels_train[1:] if labels_train is not None else None
    labels_test_aligned = labels_test[1:] if labels_test is not None else None

    label_stats_train = label_composition(labels_train_aligned, gamma_train) if labels_train_aligned is not None else None
    label_stats_test = label_composition(labels_test_aligned, gamma_test) if labels_test_aligned is not None else None

    tradeable = tradeable_states(label_stats_test) if label_stats_test is not None else []
    skip_rate_test = overall_skip_rate(label_stats_test)

    mean_diag = float(np.mean(np.diag(model.A)))
    expected_duration = []
    for i in range(k):
        diag = model.A[i, i]
        if diag >= 1.0:
            expected_duration.append(float("inf"))
        else:
            expected_duration.append(float(1.0 / max(1e-6, (1.0 - diag))))

    warnings = []
    min_neff = float(np.min(n_eff_test)) if n_eff_test.size else 0.0
    if min_neff < max(30.0, 0.05 * (x_test.shape[0] - 1)):
        warnings.append("ghost")
    if float(np.max(occ_test)) > 0.90:
        warnings.append("collapse")
    if mean_diag > 0.99:
        warnings.append("too-sticky")
    if ent.get("p90", 0.0) > 0.85:
        warnings.append("high-ambiguity")
    if eps_bumped:
        warnings.append(f"eps_bumped:{eps_bumped[0]}->{eps_bumped[1]}")

    if not warnings:
        warnings = ["OK"]

    result = {
        "status": "ok",
        "fold": fold.name,
        "train_years": fold.train_years,
        "test_years": fold.test_years,
        "featureset": None,
        "features": features,
        "n_features": int(x_train.shape[1]),
        "corr_threshold": float(corr_threshold),
        "n_features_after_prune": int(n_pruned_pool),
        "K": int(k),
        "eps_requested": float(eps),
        "eps_used": float(eps_used),
        "seed": int(seed),
        "n_train": int(x_train.shape[0]),
        "n_test": int(x_test.shape[0]),
        "fit": {
            "train_loglik": float(train_loglik),
            "test_loglik": float(test_loglik),
            "train_avg_ll": float(train_loglik / max(1, x_train.shape[0] - 1)),
            "test_avg_ll": float(test_loglik / max(1, x_test.shape[0] - 1)),
            "aic": float(aic),
            "bic": float(bic),
            "converged": bool(fit_result.converged),
            "n_iter": int(fit_result.n_iter),
        },
        "regime": {
            "A": model.A.tolist(),
            "mean_diag_A": float(mean_diag),
            "expected_duration": expected_duration,
            "switches_viterbi": int(switches),
            "occ_train": occ_train.tolist(),
            "occ_test": occ_test.tolist(),
            "n_eff_train": n_eff_train.tolist(),
            "n_eff_test": n_eff_test.tolist(),
            "entropy_test": ent,
        },
        "labels": {
            "label_col": label_col,
            "train": label_stats_train,
            "test": label_stats_test,
            "skip_rate_test": skip_rate_test,
            "tradeable_states": tradeable,
        },
        "warnings": warnings,
    }

    return result, None


def print_label_table(label_stats: Optional[Dict], title: str) -> None:
    if label_stats is None:
        print(f"{title}: label column not found")
        return
    hard = label_stats.get("hard", {})
    parts = []
    for state in sorted(hard.keys()):
        stats = hard[state]
        pct = stats["pct"]
        n = stats["total"]
        parts.append(
            f"S{state} L={format_pct(pct['long'])} S={format_pct(pct['short'])} K={format_pct(pct['skip'])} N={int(n)}"
        )
    print(f"{title}: " + " | ".join(parts))


def _safe_pct(count: float, total: float) -> float:
    if total <= 0:
        return 0.0
    return float(count) / float(total) * 100.0


def overall_skip_rate(label_stats: Optional[Dict]) -> Optional[float]:
    if label_stats is None:
        return None
    hard = label_stats.get("hard", {})
    skip_total = 0.0
    total = 0.0
    for stats in hard.values():
        counts = stats.get("counts", {})
        total += float(stats.get("total", 0.0))
        skip_total += float(counts.get("skip", 0.0))
    if total <= 0:
        return None
    return skip_total / total * 100.0


def parse_sizes(value: Optional[str]) -> Optional[List[int]]:
    if value is None:
        return None
    if not value.strip():
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    sizes = []
    for part in parts:
        try:
            sizes.append(int(part))
        except ValueError as exc:
            raise ValueError(f"Invalid size '{part}' in --search-sizes") from exc
    return sizes or None


def write_report(summary_path: str, report_xlsx: str) -> None:
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Summary JSONL not found: {summary_path}")

    rows: List[Dict] = []
    with open(summary_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("status") != "ok":
                continue
            labels = (record.get("labels") or {}).get("test")
            if not labels:
                continue
            hard = labels.get("hard", {})
            fit = record.get("fit", {})
            regime = record.get("regime", {})
            for state, stats in hard.items():
                counts = stats.get("counts", {})
                total = float(stats.get("total", 0.0))
                long_c = float(counts.get("long", 0.0))
                short_c = float(counts.get("short", 0.0))
                skip_c = float(counts.get("skip", 0.0))
                trades = long_c + short_c
                tradeable_share = _safe_pct(trades, total)

                rows.append(
                    {
                        "Period": record.get("fold"),
                        "Model ID": f"{record.get('featureset')}|eps{record.get('eps_used')}|seed{record.get('seed')}",
                        "K": record.get("K"),
                        "Cluster Number": int(state),
                        "Long %": _safe_pct(long_c, total),
                        "Short %": _safe_pct(short_c, total),
                        "Trades": trades,
                        "Skip %": _safe_pct(skip_c, total),
                        "Total": total,
                        "Corr Threshold": record.get("corr_threshold"),
                        "N Features After Prune": record.get("n_features_after_prune"),
                        "Tradeable Share": tradeable_share,
                        "Tradeable": tradeable_share >= 60.0,
                        "Featureset": record.get("featureset"),
                        "N Features": record.get("n_features"),
                        "Test Avg LL": fit.get("test_avg_ll"),
                        "Train Avg LL": fit.get("train_avg_ll"),
                        "Mean Diag A": regime.get("mean_diag_A"),
                        "Switches": regime.get("switches_viterbi"),
                        "Eps": record.get("eps_used"),
                        "Seed": record.get("seed"),
                    }
                )

    if not rows:
        raise ValueError("No label stats found in summary; cannot build report.")

    df = pd.DataFrame(rows)
    df = df.sort_values(["Period", "Model ID", "K", "Cluster Number"]).reset_index(drop=True)

    try:
        import openpyxl  # noqa: F401

        df.to_excel(report_xlsx, index=False, engine="openpyxl")
    except Exception:
        report_csv = os.path.splitext(report_xlsx)[0] + ".csv"
        df.to_csv(report_csv, index=False)
        raise RuntimeError(f"Excel writer unavailable; wrote CSV instead: {report_csv}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--smoke", action="store_true", help="Run a single fold/config for a smoke test")
    parser.add_argument(
        "--feature-search",
        action="store_true",
        help="Run feature subsets with a reduced hyperparameter grid (K list, single eps/seed).",
    )
    parser.add_argument(
        "--fold",
        type=str,
        default=None,
        help="Optional fold name filter, e.g. 2020_2024_test_2025",
    )
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--use-ledoitwolf", action="store_true")
    parser.add_argument("--corr-threshold", type=float, default=CORR_THRESHOLD)
    parser.add_argument(
        "--search-skip-target",
        type=float,
        default=None,
        help="Random feature search until overall test skip%% falls below this value.",
    )
    parser.add_argument(
        "--search-max-trials",
        type=int,
        default=20,
        help="Max random feature subsets to try when --search-skip-target is set.",
    )
    parser.add_argument(
        "--search-sizes",
        type=str,
        default=None,
        help="Comma-separated feature subset sizes for skip search (e.g. 8,12,16,20).",
    )
    parser.add_argument(
        "--search-use-unpruned",
        action="store_true",
        help="Use the unpruned feature pool for skip search.",
    )
    parser.add_argument(
        "--search-allow-repeat-subsets",
        action="store_true",
        help="Allow repeated subsets during skip search (resampling with replacement).",
    )
    parser.add_argument(
        "--report-xlsx",
        type=str,
        default=os.path.join(RESULTS_DIR, "arhmm_report.xlsx"),
        help="Write Excel report from summary JSONL (per-state label stats).",
    )
    args = parser.parse_args()

    ensure_dir(RESULTS_DIR)

    df = load_data(args.data_path)
    label_col = detect_label_column(df)

    candidates = indicator_columns(df, label_col)
    if not candidates:
        raise ValueError("No indicator columns found after exclusions.")

    recommended_paths = [
        os.path.join("pipeline", "step_feature_select", "results", "recommended_features.csv"),
        os.path.join("pipeline", "step_feature_select", "results", "top_features.csv"),
        os.path.join("pipeline", "step_feature_select", "recommended_features.csv"),
    ]
    recommended = load_recommended_features(recommended_paths)
    source = "file"
    if not recommended:
        recommended = variance_fallback_features(df, candidates, top_n=20)
        source = "variance"

    recommended = [c for c in recommended if c in candidates]
    if not recommended:
        raise ValueError("Recommended features list is empty after filtering to indicator columns.")

    folds = build_folds(df)
    if not folds:
        raise ValueError("No valid folds available within the requested date range.")

    if args.fold:
        folds = [f for f in folds if f.name == args.fold]
        if not folds:
            raise ValueError(f"No fold matched '{args.fold}'. Available: {[f.name for f in build_folds(df)]}")

    if args.smoke:
        folds = folds[:1]
        k_list = [K_LIST[0]]
        eps_list = [EPS_LIST[0]]
        seed_list = [SEED_LIST[0]]
    elif args.feature_search:
        k_list = K_LIST
        eps_list = [EPS_LIST[0]]
        seed_list = [SEED_LIST[0]]
    else:
        k_list = K_LIST
        eps_list = EPS_LIST
        seed_list = SEED_LIST

    if args.search_skip_target is not None and label_col is None:
        raise ValueError("search-skip-target requires a label column (candle_type or label).")

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        fold_feature_counts: Dict[str, Tuple[int, int]] = {}
        for fold in folds:
            train_df, test_df = split_fold(df, fold)
            train_matrix = train_df[recommended]
            non_all_nan = [col for col in recommended if not train_matrix[col].isna().all()]
            if not non_all_nan:
                raise ValueError(f"Fold {fold.name}: all recommended features are NaN.")

            x_train = train_matrix[non_all_nan].values.astype(float)
            pruned_pool = prune_correlated_features(
                x_train=x_train,
                feature_names=non_all_nan,
                corr_threshold=args.corr_threshold,
            )
            fold_feature_counts[fold.name] = (len(non_all_nan), len(pruned_pool))

            corr_tag = f"corr{int(args.corr_threshold * 100):03d}"
            fold_dir = os.path.join(RESULTS_DIR, fold.name)
            ensure_dir(fold_dir)
            pruned_path = os.path.join(fold_dir, f"pruned_features_{corr_tag}.csv")
            pd.DataFrame({"feature": pruned_pool}).to_csv(pruned_path, index=False)

            if args.search_skip_target is not None:
                search_sizes = parse_sizes(args.search_sizes) or SUBSET_SIZES
                pool = non_all_nan if args.search_use_unpruned else pruned_pool
                feature_sets = OrderedDict(
                    build_random_feature_sets(
                        pool,
                        search_sizes,
                        args.search_max_trials,
                        allow_repeat_subsets=args.search_allow_repeat_subsets,
                    )
                )
            elif args.smoke:
                feature_sets = OrderedDict(list(build_feature_sets(pruned_pool, prefix="rec_pruned").items())[:1])
            else:
                feature_sets = build_feature_sets(pruned_pool, prefix="rec_pruned")

            found_skip_target = False
            for featureset_name, features in feature_sets.items():
                for k in k_list:
                    for eps in eps_list:
                        for seed in seed_list:
                            result, error = run_one_config(
                                fold=fold,
                                train_df=train_df,
                                test_df=test_df,
                                features=features,
                                label_col=label_col,
                                k=k,
                                eps=eps,
                                seed=seed,
                                max_iter=args.max_iter,
                                tol=args.tol,
                                use_ledoitwolf=args.use_ledoitwolf,
                                eps_list=EPS_LIST,
                                corr_threshold=args.corr_threshold,
                                n_pruned_pool=len(pruned_pool) if not args.search_use_unpruned else len(non_all_nan),
                            )

                            if result is None:
                                record = {
                                    "status": "failed",
                                    "fold": fold.name,
                                    "train_years": fold.train_years,
                                    "test_years": fold.test_years,
                                    "featureset": featureset_name,
                                    "features": features,
                                    "n_features": len(features),
                                    "corr_threshold": float(args.corr_threshold),
                                    "n_features_after_prune": int(len(pruned_pool) if not args.search_use_unpruned else len(non_all_nan)),
                                    "K": int(k),
                                    "eps_requested": float(eps),
                                    "seed": int(seed),
                                    "error": error or "unknown",
                                }
                                f.write(json.dumps(record) + "\n")
                                print(
                                    f"ARHMM | fold={fold.name} | featureset={featureset_name} | K={k} | eps={eps} | seed={seed} "
                                    f"| n_train=0 n_test=0 n_feat={len(features)}"
                                )
                                print(f"FIT {{train_ll:NA, test_ll:NA, AIC:NA, BIC:NA}}")
                                print("REGIME {mean_diag_A:NA, switches_viterbi:NA, occ_test:NA, N_eff_test_min:NA, ent_p50:NA, ent_p90:NA}")
                                print("LABEL_STATE_TABLE: NA")
                                print(f"WARNINGS: failed:{error or 'unknown'}")
                                continue

                            result["featureset"] = featureset_name
                            f.write(json.dumps(result) + "\n")

                            fit = result["fit"]
                            regime = result["regime"]

                            print(
                                f"ARHMM | fold={fold.name} | featureset={featureset_name} | K={k} | eps={result['eps_used']} | seed={seed} "
                                f"| n_train={result['n_train']} n_test={result['n_test']} n_feat={result['n_features']}"
                            )
                            print(
                                "FIT {"
                                f"train_ll:{fit['train_avg_ll']:.6f}, test_ll:{fit['test_avg_ll']:.6f}, AIC:{fit['aic']:.2f}, BIC:{fit['bic']:.2f}"
                                "}"
                            )
                            print(
                                "REGIME {"
                                f"mean_diag_A:{regime['mean_diag_A']:.4f}, switches_viterbi:{regime['switches_viterbi']}, "
                                f"occ_test:{[round(x, 4) for x in regime['occ_test']]}, "
                                f"N_eff_test_min:{min(regime['n_eff_test']):.1f}, "
                                f"ent_p50:{regime['entropy_test']['p50']:.3f}, ent_p90:{regime['entropy_test']['p90']:.3f}"
                                "}"
                            )
                            print_label_table(result["labels"]["test"], "LABEL_STATE_TABLE")

                            tradeable = result["labels"]["tradeable_states"]
                            if tradeable:
                                trade_parts = [f"S{state}:{score:.2f}" for state, score in tradeable]
                                print("TRADEABLE: " + ", ".join(trade_parts))

                                print("WARNINGS: " + ", ".join(result["warnings"]))

                            if args.search_skip_target is not None:
                                skip_rate = result["labels"].get("skip_rate_test")
                                if skip_rate is not None and skip_rate < args.search_skip_target:
                                    print(
                                        f"FOUND skip< {args.search_skip_target}%: featureset={featureset_name} "
                                        f"K={k} eps={result['eps_used']} seed={seed} skip={skip_rate:.2f}%"
                                    )
                                    found_skip_target = True
                                    break
                        if found_skip_target:
                            break
                    if found_skip_target:
                        break
                if found_skip_target:
                    break
            if args.search_skip_target is not None and not found_skip_target:
                print(
                    f"No subset met skip < {args.search_skip_target}% within {len(feature_sets)} trials for fold {fold.name}."
                )

    print(f"Summary written to {SUMMARY_PATH}")
    print(f"Recommended feature source: {source}")
    if folds:
        for fold in folds:
            if fold.name in fold_feature_counts:
                before_count, after_count = fold_feature_counts[fold.name]
                print(
                    f"Fold {fold.name}: features before prune={before_count} after prune={after_count} (threshold={args.corr_threshold})"
                )
    if not SKLEARN_AVAILABLE:
        print("Note: sklearn not available; using numpy scaling.")
    if args.report_xlsx:
        try:
            write_report(SUMMARY_PATH, args.report_xlsx)
            print(f"Excel report written to {args.report_xlsx}")
        except Exception as exc:
            print(f"Report generation failed: {exc}")
    try:
        tradeable_rows = []
        with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("status") != "ok":
                    continue
                labels = (record.get("labels") or {}).get("test")
                if not labels:
                    continue
                hard = labels.get("hard", {})
                for state, stats in hard.items():
                    counts = stats.get("counts", {})
                    total = float(stats.get("total", 0.0))
                    long_c = float(counts.get("long", 0.0))
                    short_c = float(counts.get("short", 0.0))
                    skip_c = float(counts.get("skip", 0.0))
                    trades = long_c + short_c
                    tradeable_share = _safe_pct(trades, total)
                    if total < 150 or tradeable_share < 55.0:
                        continue
                    tradeable_rows.append(
                        {
                            "Period": record.get("fold"),
                            "Model ID": f"{record.get('featureset')}|eps{record.get('eps_used')}|seed{record.get('seed')}",
                            "K": record.get("K"),
                            "Cluster": int(state),
                            "Long %": _safe_pct(long_c, total),
                            "Short %": _safe_pct(short_c, total),
                            "Trades": trades,
                            "Skip %": _safe_pct(skip_c, total),
                            "Total": total,
                            "Tradeable Share": tradeable_share,
                        }
                    )
        if tradeable_rows:
            df_tradeable = pd.DataFrame(tradeable_rows)
            df_tradeable = df_tradeable.sort_values(
                ["Tradeable Share", "Trades"], ascending=[False, False]
            ).head(10)
            print("\nTop tradeable regimes (Total>=150, Tradeable Share>=55):")
            print(df_tradeable.to_string(index=False))
        else:
            print("\nTop tradeable regimes: none matched Total>=150 and Tradeable Share>=55.")
    except Exception as exc:
        print(f"Tradeable summary failed: {exc}")


if __name__ == "__main__":
    main()

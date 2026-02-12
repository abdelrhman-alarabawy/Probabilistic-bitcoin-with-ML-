#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEED = 42
TRAIN_FRAC = 0.80
VALID_FRAC = 0.10
TEST_FRAC = 0.10
R = 200
TOPK = 20
NEG_POS_RATIO = 5
CORR_THRESHOLD = 0.95
STABILITY_MIN = 0.30
KS_MIN = 0.10
MAX_ITER = 8000

DATA_PATH = Path("data/external/output_1d_labels.csv")
OUT_ROOT = Path("reports/feature_discovery")
PLOTS_DIR = OUT_ROOT / "plots"
RANKINGS_DIR = OUT_ROOT / "rankings"
REGIME_DIR = OUT_ROOT / "regimes"

LABEL_CANDIDATES = ["label", "signal", "target", "y", "class", "trade_label", "candle_type"]
TIME_PATTERNS = ["timestamp", "time", "date", "datetime", "local_"]
OHLCV_PATTERNS = ["open", "high", "low", "close", "volume", "ohlcv"]
OHLCV_EXACT_ALIASES = {
    "open", "high", "low", "close", "volume", "o", "h", "l", "c", "v", "adj_close", "adjclose"
}


def ensure_dirs() -> None:
    for d in [OUT_ROOT, PLOTS_DIR, RANKINGS_DIR, REGIME_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def detect_label_column(df: pd.DataFrame) -> str:
    lower_map = {c.lower(): c for c in df.columns}
    hits = [lower_map[c] for c in LABEL_CANDIDATES if c in lower_map]
    if not hits:
        hits = [c for c in df.columns if any(k in c.lower() for k in LABEL_CANDIDATES)]

    def score_col(col: str) -> Tuple[int, int]:
        vals = df[col].dropna().astype(str).str.lower().str.strip()
        uniq = set(vals.unique().tolist())
        score = len(uniq.intersection({"long", "short", "skip", "buy", "sell", "hold", "flat", "0", "1", "-1"}))
        return score, len(uniq)

    if hits:
        ranked = sorted(hits, key=lambda c: score_col(c), reverse=True)
        chosen = ranked[0]
    else:
        candidates = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype).startswith("int")]
        if not candidates:
            raise ValueError("Could not find any plausible label column.")
        ranked = sorted(candidates, key=lambda c: score_col(c), reverse=True)
        chosen = ranked[0]

    print(f"Label candidates: {hits if hits else 'fallback scan'}")
    print(f"Chosen label column: {chosen}")
    return chosen


def canonicalize_labels(s: pd.Series) -> pd.Series:
    ss = s.astype(str).str.lower().str.strip()
    mapping = {
        "long": "long", "buy": "long", "1": "long", "bull": "long",
        "short": "short", "sell": "short", "-1": "short", "bear": "short",
        "skip": "skip", "hold": "skip", "flat": "skip", "0": "skip", "none": "skip",
    }
    mapped = ss.map(mapping)
    if mapped.isna().any():
        uniq = sorted(ss.dropna().unique().tolist())
        if len(uniq) == 3:
            auto_map = {uniq[0]: "long", uniq[1]: "short", uniq[2]: "skip"}
            mapped = ss.map(auto_map)
            print(f"Auto-mapped 3 labels by lexicographic order: {auto_map}")
        else:
            mapped = mapped.fillna("skip")
    return mapped


def is_time_col(name: str) -> bool:
    n = name.lower()
    return any(p in n for p in TIME_PATTERNS)


def is_ohlcv_col(name: str) -> bool:
    n = name.lower()
    return n in OHLCV_EXACT_ALIASES or any(p in n for p in OHLCV_PATTERNS)


def chrono_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    i1 = int(n * TRAIN_FRAC)
    i2 = int(n * (TRAIN_FRAC + VALID_FRAC))
    train, valid, test = df.iloc[:i1].copy(), df.iloc[i1:i2].copy(), df.iloc[i2:].copy()
    return train, valid, test


def safe_ap(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if y_true.sum() == 0:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


def cohen_d(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) < 2 or len(neg) < 2:
        return 0.0
    var1, var2 = np.var(pos, ddof=1), np.var(neg, ddof=1)
    pooled = ((len(pos) - 1) * var1 + (len(neg) - 1) * var2) / max((len(pos) + len(neg) - 2), 1)
    if pooled <= 1e-12:
        return 0.0
    return float((np.mean(pos) - np.mean(neg)) / np.sqrt(pooled))


def rank_from_values(values: pd.Series, descending: bool = True) -> pd.Series:
    return values.rank(method="average", ascending=not descending)


def plot_top_bars(df_rank: pd.DataFrame, score_col: str, title: str, out_path: Path, topn: int = 20) -> None:
    dd = df_rank.sort_values(score_col, ascending=True).head(topn).iloc[::-1]
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(dd["feature"], dd[score_col], color="#2f7ed8")
    ax.set_title(title)
    ax.set_xlabel(score_col)
    plt.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_distributions(X_train: pd.DataFrame, y_train_bin: np.ndarray, features: List[str], title: str, out_path: Path) -> None:
    k = min(5, len(features))
    if k == 0:
        return
    fig, axes = plt.subplots(k, 1, figsize=(12, 2.6 * k), squeeze=False)
    pos = y_train_bin == 1
    neg = y_train_bin == 0
    for i, f in enumerate(features[:k]):
        ax = axes[i, 0]
        ax.hist(X_train.loc[neg, f], bins=40, alpha=0.55, density=True, label="neg")
        ax.hist(X_train.loc[pos, f], bins=40, alpha=0.55, density=True, label="pos")
        ax.set_title(f)
        ax.legend(loc="upper right")
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def select_regime_features(X_train: pd.DataFrame, max_features: int = 12) -> List[str]:
    kw = ["atr", "tr", "vol", "bb_width", "adx", "volume_roc", "volume_trend", "returns_vol", "range", "drawdown"]
    missing = X_train.isna().mean()
    var = X_train.var(numeric_only=True)
    pool = pd.DataFrame({"feature": X_train.columns})
    pool["missing"] = pool["feature"].map(missing).fillna(1.0)
    pool["var"] = pool["feature"].map(var).fillna(0.0)
    pool["pref"] = pool["feature"].str.lower().apply(lambda x: int(any(k in x for k in kw)))
    pool = pool.sort_values(["pref", "var", "missing"], ascending=[False, False, True]).reset_index(drop=True)

    chosen: List[str] = []
    for feat in pool["feature"].tolist():
        if len(chosen) >= max_features:
            break
        if not chosen:
            chosen.append(feat)
            continue
        c = X_train[chosen + [feat]].corr(method="spearman").abs()[feat].drop(feat)
        if (c <= 0.90).all() or c.empty:
            chosen.append(feat)
    return chosen[:max_features]


def discover_for_class(
    cls: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
    plots_dir: Path,
    stability_repeats: int = R,
    perm_repeats_logit: int = 6,
    perm_repeats_tree: int = 5,
) -> Dict[str, object]:
    y_tr = (y_train == cls).astype(int).to_numpy()
    y_va = (y_valid == cls).astype(int).to_numpy()
    y_te = (y_test == cls).astype(int).to_numpy()

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X_train)
    Xva_s = scaler.transform(X_valid)
    Xte_s = scaler.transform(X_test)

    c_grid = [0.01, 0.1, 1, 10]
    best_c, best_ap = c_grid[0], -1.0
    best_model = None
    for c in c_grid:
        model = LogisticRegression(
            solver="saga", l1_ratio=0.5, C=c,
            class_weight="balanced", max_iter=MAX_ITER, random_state=SEED,
            tol=1e-3,
        )
        model.fit(Xtr_s, y_tr)
        p = model.predict_proba(Xva_s)[:, 1]
        ap = safe_ap(y_va, p)
        if np.isnan(ap):
            ap = -1.0
        if ap > best_ap:
            best_ap = ap
            best_c = c
            best_model = model

    assert best_model is not None
    logit_coef_abs = pd.Series(np.abs(best_model.coef_[0]), index=X_train.columns)

    perm = permutation_importance(
        best_model, Xva_s, y_va, scoring="average_precision", n_repeats=perm_repeats_logit,
        random_state=SEED, n_jobs=-1,
    )
    perm_importance = pd.Series(perm.importances_mean, index=X_train.columns)

    mi = pd.Series(mutual_info_classif(X_train, y_tr, discrete_features=False, random_state=SEED), index=X_train.columns)

    pos_idx = y_tr == 1
    ks_vals, d_vals = {}, {}
    for f in X_train.columns:
        pos = X_train.loc[pos_idx, f].to_numpy()
        neg = X_train.loc[~pos_idx, f].to_numpy()
        ks_vals[f] = float(ks_2samp(pos, neg).statistic) if len(pos) > 5 and len(neg) > 5 else 0.0
        d_vals[f] = abs(cohen_d(pos, neg))

    rf = RandomForestClassifier(
        n_estimators=180, random_state=SEED, n_jobs=-1,
        class_weight="balanced_subsample", min_samples_leaf=3,
    )
    rf.fit(X_train, y_tr)
    rf_perm = permutation_importance(
        rf, X_valid, y_va, scoring="average_precision", n_repeats=perm_repeats_tree,
        random_state=SEED, n_jobs=-1,
    )
    tree_perm = pd.Series(rf_perm.importances_mean, index=X_train.columns)

    # Stability selection
    rng = np.random.default_rng(SEED)
    feat_names = np.array(X_train.columns)
    freq = pd.Series(0, index=X_train.columns, dtype=float)
    pos_all = np.where(y_tr == 1)[0]
    neg_all = np.where(y_tr == 0)[0]
    ratio = 10 if len(pos_all) < 300 else NEG_POS_RATIO
    for _ in range(stability_repeats):
        if len(pos_all) == 0:
            break
        need_neg = min(len(neg_all), len(pos_all) * ratio)
        neg_sub = rng.choice(neg_all, size=need_neg, replace=False)
        idx = np.concatenate([pos_all, neg_sub])
        rng.shuffle(idx)
        Xs = Xtr_s[idx]
        ys = y_tr[idx]
        m = LogisticRegression(
            solver="saga", l1_ratio=0.5, C=best_c,
            class_weight="balanced", max_iter=MAX_ITER, random_state=SEED,
            tol=1e-3,
        )
        m.fit(Xs, ys)
        top_idx = np.argsort(np.abs(m.coef_[0]))[::-1][:TOPK]
        freq.loc[feat_names[top_idx]] += 1
    stability = freq / max(stability_repeats, 1)

    rank_df = pd.DataFrame({
        "feature": X_train.columns,
        "stability_freq": stability.values,
        "ks": pd.Series(ks_vals),
        "cohen_d": pd.Series(d_vals),
        "mi": mi.values,
        "perm_importance_mean": perm_importance.values,
        "logit_coef_abs": logit_coef_abs.values,
        "tree_perm_importance_mean": tree_perm.values,
    }).fillna(0.0)

    rank_df["rank_logit"] = rank_from_values(rank_df["logit_coef_abs"], True)
    rank_df["rank_perm"] = rank_from_values(rank_df["perm_importance_mean"], True)
    rank_df["rank_mi"] = rank_from_values(rank_df["mi"], True)
    rank_df["rank_ks"] = rank_from_values(rank_df["ks"], True)
    rank_df["rank_cohen"] = rank_from_values(rank_df["cohen_d"], True)
    rank_df["rank_stability"] = rank_from_values(rank_df["stability_freq"], True)
    rank_df["rank_tree_perm"] = rank_from_values(rank_df["tree_perm_importance_mean"], True)
    rank_df["rank_fused"] = rank_df[["rank_logit", "rank_perm", "rank_mi", "rank_ks", "rank_cohen", "rank_stability", "rank_tree_perm"]].mean(axis=1)
    rank_df = rank_df.sort_values("rank_fused").reset_index(drop=True)

    trusted = rank_df[(rank_df["stability_freq"] >= STABILITY_MIN) & (rank_df["ks"] >= KS_MIN)].copy()
    trusted = trusted.sort_values("rank_fused").head(30)

    full_valid_ap = safe_ap(y_va, best_model.predict_proba(Xva_s)[:, 1])
    full_test_ap = safe_ap(y_te, best_model.predict_proba(Xte_s)[:, 1])

    trusted_valid_ap, trusted_test_ap = np.nan, np.nan
    if not trusted.empty:
        trusted_feats = trusted["feature"].tolist()
        scl_t = StandardScaler()
        Xt_tr = scl_t.fit_transform(X_train[trusted_feats])
        Xt_va = scl_t.transform(X_valid[trusted_feats])
        Xt_te = scl_t.transform(X_test[trusted_feats])
        m_t = LogisticRegression(
            solver="saga", l1_ratio=0.5, C=best_c,
            class_weight="balanced", max_iter=MAX_ITER, random_state=SEED,
            tol=1e-3,
        )
        m_t.fit(Xt_tr, y_tr)
        trusted_valid_ap = safe_ap(y_va, m_t.predict_proba(Xt_va)[:, 1])
        trusted_test_ap = safe_ap(y_te, m_t.predict_proba(Xt_te)[:, 1])

    ranking_path = output_dir / f"{cls}_rankings.csv"
    trusted_path = output_dir / f"{cls}_trusted_features.csv"
    rank_df.to_csv(ranking_path, index=False)
    trusted.to_csv(trusted_path, index=False)

    plot_top_bars(rank_df, "rank_fused", f"{cls.upper()} Top20 (lowest fused rank is best)", plots_dir / f"{cls}_top20_fused.png")
    st_df = rank_df.sort_values("stability_freq", ascending=False).head(20).copy()
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(st_df["feature"].iloc[::-1], st_df["stability_freq"].iloc[::-1], color="#f28f43")
    ax.set_title(f"{cls.upper()} Stability Top20")
    ax.set_xlabel("stability_freq")
    plt.tight_layout()
    fig.savefig(plots_dir / f"{cls}_stability_top20.png", dpi=140)
    plt.close(fig)
    plot_distributions(X_train, y_tr, trusted["feature"].tolist()[:5], f"{cls.upper()} top5 trusted feature distributions", plots_dir / f"{cls}_top5_distributions.png")

    return {
        "class": cls,
        "best_c": best_c,
        "full_valid_ap": full_valid_ap,
        "full_test_ap": full_test_ap,
        "trusted_valid_ap": trusted_valid_ap,
        "trusted_test_ap": trusted_test_ap,
        "top20_fused": rank_df.head(20)["feature"].tolist(),
        "top20_stability": rank_df.sort_values("stability_freq", ascending=False).head(20)["feature"].tolist(),
        "trusted_top20": trusted.head(20)[["feature", "stability_freq", "ks"]].to_dict(orient="records"),
        "trusted_top10": trusted.head(10)["feature"].tolist(),
    }


def run_discovery(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    rankings_dir: Path,
    plots_dir: Path,
) -> Dict[str, Dict[str, object]]:
    res: Dict[str, Dict[str, object]] = {}
    for cls in ["long", "short", "skip"]:
        res[cls] = discover_for_class(cls, X_train, y_train, X_valid, y_valid, X_test, y_test, rankings_dir, plots_dir)
    return res


def main() -> None:
    np.random.seed(SEED)
    ensure_dirs()

    print(f"Loading: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print("Shape:", df.shape)
    print(df.head(2))
    print("Dtypes summary:\n", df.dtypes.value_counts())

    label_col = detect_label_column(df)
    df["label_canonical"] = canonicalize_labels(df[label_col])
    print("Overall label counts:\n", df["label_canonical"].value_counts(dropna=False))

    dropped_time = [c for c in df.columns if is_time_col(c)]
    dropped_ohlcv = [c for c in df.columns if is_ohlcv_col(c)]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feat_initial = [c for c in numeric_cols if c not in dropped_ohlcv and c not in dropped_time and c != label_col]

    train_df, valid_df, test_df = chrono_split(df)

    # Missingness diagnostics
    miss_before = df[feat_initial].isna().mean().mean() if feat_initial else 0.0

    # ffill without shifting
    for part in (train_df, valid_df, test_df):
        part[feat_initial] = part[feat_initial].ffill()

    # constant/near-constant removal on train
    variances = train_df[feat_initial].var(numeric_only=True)
    feat_const_removed = [f for f in feat_initial if variances.get(f, 0.0) >= 1e-12]

    # correlation prune on train
    corr = train_df[feat_const_removed].corr(method="spearman").abs()
    to_drop = set()
    cols = feat_const_removed
    for i in range(len(cols)):
        if cols[i] in to_drop:
            continue
        for j in range(i + 1, len(cols)):
            if corr.iloc[i, j] > CORR_THRESHOLD:
                to_drop.add(cols[j])
    feat_final = [f for f in feat_const_removed if f not in to_drop]

    # median impute on train only
    imp = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(imp.fit_transform(train_df[feat_final]), columns=feat_final, index=train_df.index)
    X_valid = pd.DataFrame(imp.transform(valid_df[feat_final]), columns=feat_final, index=valid_df.index)
    X_test = pd.DataFrame(imp.transform(test_df[feat_final]), columns=feat_final, index=test_df.index)
    miss_after = pd.concat([X_train, X_valid, X_test], axis=0).isna().mean().mean()

    y_train = train_df["label_canonical"]
    y_valid = valid_df["label_canonical"]
    y_test = test_df["label_canonical"]

    print(f"Initial features: {len(feat_initial)}")
    print(f"After constant removal: {len(feat_const_removed)}")
    print(f"After corr prune: {len(feat_final)}")
    print(f"Missingness before: {miss_before:.6f}; after: {miss_after:.6f}")

    if any("timestamp" in c.lower() for c in df.columns):
        ts_col = [c for c in df.columns if "timestamp" in c.lower()][0]
        print("Train range:", train_df[ts_col].iloc[0], "->", train_df[ts_col].iloc[-1])
        print("Valid range:", valid_df[ts_col].iloc[0], "->", valid_df[ts_col].iloc[-1])
        print("Test range:", test_df[ts_col].iloc[0], "->", test_df[ts_col].iloc[-1])
    else:
        print("Train index range:", train_df.index.min(), train_df.index.max())
        print("Valid index range:", valid_df.index.min(), valid_df.index.max())
        print("Test index range:", test_df.index.min(), test_df.index.max())

    print("Split label counts:")
    print("Train:\n", y_train.value_counts())
    print("Valid:\n", y_valid.value_counts())
    print("Test:\n", y_test.value_counts())

    global_res = run_discovery(X_train, y_train, X_valid, y_valid, X_test, y_test, RANKINGS_DIR, PLOTS_DIR)

    # Global PR-AUC plot
    pr_rows = []
    for cls, r in global_res.items():
        pr_rows.extend([
            {"class": cls, "model": "full_valid", "ap": r["full_valid_ap"]},
            {"class": cls, "model": "full_test", "ap": r["full_test_ap"]},
            {"class": cls, "model": "trusted_valid", "ap": r["trusted_valid_ap"]},
            {"class": cls, "model": "trusted_test", "ap": r["trusted_test_ap"]},
        ])
    pr_df = pd.DataFrame(pr_rows)
    fig, ax = plt.subplots(figsize=(10, 6))
    piv = pr_df.pivot(index="class", columns="model", values="ap").fillna(0)
    piv.plot(kind="bar", ax=ax)
    ax.set_ylabel("PR-AUC")
    ax.set_title("Global PR-AUC by class/model")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "global_pr_auc.png", dpi=140)
    plt.close(fig)

    # Regime discovery
    regime_features = select_regime_features(X_train, max_features=12)
    scaler_reg = StandardScaler()
    Z_train = scaler_reg.fit_transform(X_train[regime_features])

    bics: Dict[int, float] = {}
    gm_models: Dict[int, GaussianMixture] = {}
    for k in [3, 4, 5, 6]:
        gm = GaussianMixture(n_components=k, random_state=SEED, covariance_type="full", reg_covar=1e-6)
        gm.fit(Z_train)
        bics[k] = float(gm.bic(Z_train))
        gm_models[k] = gm
    best_k = min(bics, key=bics.get)
    gm = gm_models[best_k]

    Z_all = scaler_reg.transform(pd.concat([X_train[regime_features], X_valid[regime_features], X_test[regime_features]], axis=0))
    regime_ids = gm.predict(Z_all)
    full_idx = pd.concat([X_train, X_valid, X_test], axis=0).index
    regime_assign = pd.DataFrame({"row_index": full_idx, "regime_id": regime_ids})
    regime_assign.to_csv(REGIME_DIR / "regime_assignments.csv", index=False)

    regime_summary = {
        "best_k": int(best_k),
        "bics": {str(k): v for k, v in bics.items()},
        "regime_features": regime_features,
        "regime_sizes": regime_assign["regime_id"].value_counts().sort_index().to_dict(),
    }
    (REGIME_DIR / "regime_model_summary.json").write_text(json.dumps(regime_summary, indent=2), encoding="utf-8")

    split_name = pd.Series(index=full_idx, dtype=object)
    split_name.loc[X_train.index] = "train"
    split_name.loc[X_valid.index] = "valid"
    split_name.loc[X_test.index] = "test"

    all_X = pd.concat([X_train, X_valid, X_test], axis=0)
    all_y = pd.concat([y_train, y_valid, y_test], axis=0)
    all_reg = pd.Series(regime_ids, index=full_idx)

    regime_results: Dict[str, Dict[str, object]] = {}
    for r in sorted(np.unique(regime_ids)):
        rdir = REGIME_DIR / f"regime_{r}"
        rdir.mkdir(parents=True, exist_ok=True)
        mask = all_reg == r
        xr, yr, sr = all_X.loc[mask], all_y.loc[mask], split_name.loc[mask]
        tr_mask, va_mask, te_mask = sr == "train", sr == "valid", sr == "test"

        if tr_mask.sum() < 200 or va_mask.sum() < 30 or te_mask.sum() < 30:
            for cls in ["long", "short", "skip"]:
                pd.DataFrame(columns=["feature", "note"]).to_csv(rdir / f"{cls}_rankings.csv", index=False)
                pd.DataFrame(columns=["feature", "note"]).to_csv(rdir / f"{cls}_trusted_features.csv", index=False)
            regime_results[str(r)] = {"note": "insufficient rows by split"}
            continue

        rr: Dict[str, object] = {}
        for cls in ["long", "short", "skip"]:
            if (yr[tr_mask] == cls).sum() < 80:
                pd.DataFrame([{"feature": "", "note": "insufficient positives (<80)"}]).to_csv(rdir / f"{cls}_rankings.csv", index=False)
                pd.DataFrame([{"feature": "", "note": "insufficient positives (<80)"}]).to_csv(rdir / f"{cls}_trusted_features.csv", index=False)
                rr[cls] = {"note": "insufficient positives"}
                continue
            res = discover_for_class(
                cls,
                xr.loc[tr_mask], yr.loc[tr_mask],
                xr.loc[va_mask], yr.loc[va_mask],
                xr.loc[te_mask], yr.loc[te_mask],
                rdir, rdir,
                stability_repeats=60,
                perm_repeats_logit=4,
                perm_repeats_tree=3,
            )
            rr[cls] = res
        regime_results[str(r)] = rr

    # report markdown
    report = []
    report.append("# Feature Discovery Report\n")
    report.append("## 1) Dataset summary\n")
    report.append(f"- Path: `{DATA_PATH}`\n")
    report.append(f"- Rows: {len(df)}\n")
    report.append(f"- Overall label counts: `{df['label_canonical'].value_counts().to_dict()}`\n")
    report.append(f"- Train label counts: `{y_train.value_counts().to_dict()}`\n")
    report.append(f"- Valid label counts: `{y_valid.value_counts().to_dict()}`\n")
    report.append(f"- Test label counts: `{y_test.value_counts().to_dict()}`\n")

    report.append("\n## 2) Feature preparation\n")
    report.append(f"- Excluded OHLCV columns: `{dropped_ohlcv}`\n")
    report.append(f"- Excluded time columns: `{dropped_time}`\n")
    report.append(f"- Feature counts: initial={len(feat_initial)}, after_constant={len(feat_const_removed)}, after_corr={len(feat_final)}\n")
    report.append(f"- Missingness mean: before={miss_before:.6f}, after={miss_after:.6f}\n")

    report.append("\n## 3) Global results\n")
    for cls in ["long", "short", "skip"]:
        r = global_res[cls]
        report.append(f"### {cls.upper()}\n")
        report.append(f"- PR-AUC full: valid={r['full_valid_ap']:.4f}, test={r['full_test_ap']:.4f}\n")
        report.append(f"- PR-AUC trusted: valid={r['trusted_valid_ap']:.4f}, test={r['trusted_test_ap']:.4f}\n")
        report.append(f"- Top 20 fused: {r['top20_fused']}\n")
        report.append(f"- Top 20 stability: {r['top20_stability']}\n")
        report.append(f"- Trusted top 20 (feature, stability, ks): {r['trusted_top20']}\n")

    report.append("\n## 4) Regime results\n")
    report.append(f"- Chosen K by BIC: **{best_k}**\n")
    report.append(f"- BIC table: `{bics}`\n")
    report.append(f"- Regime sizes: `{regime_summary['regime_sizes']}`\n")
    for rid, rr in regime_results.items():
        report.append(f"### Regime {rid}\n")
        report.append(f"- Summary: `{rr}`\n")

    improved = False
    for cls in ["long", "short"]:
        glob = global_res[cls].get("trusted_valid_ap", np.nan)
        best_reg = -np.inf
        for rr in regime_results.values():
            if isinstance(rr, dict) and cls in rr and isinstance(rr[cls], dict):
                v = rr[cls].get("trusted_valid_ap", np.nan)
                if pd.notna(v):
                    best_reg = max(best_reg, float(v))
        if np.isfinite(best_reg) and pd.notna(glob) and best_reg > float(glob):
            improved = True
    report.append(f"- Regime conditioning improved separation for long/short: **{'Yes' if improved else 'No'}**\n")

    report.append("\n## 5) Recommendations\n")
    report.append("- Start with global trusted features (top-ranked by fused rank and stability gate).\n")
    report.append("- Consider separate per-regime models only if regime trusted PR-AUC consistently beats global.\n")
    report.append("- If class separation remains weak, refine labels or adjust prediction horizon.\n")

    (OUT_ROOT / "feature_discovery_report.md").write_text("".join(report), encoding="utf-8")

    print("\n==== FINAL SUMMARY ====")
    print("Label counts:")
    print(df["label_canonical"].value_counts())
    print("\nPR-AUC table:")
    print(pr_df)
    for cls in ["long", "short", "skip"]:
        print(f"Top 10 trusted {cls}: {global_res[cls]['trusted_top10']}")
    print(f"Regime conditioning improved PR-AUC for long/short: {'Yes' if improved else 'No'}")


if __name__ == "__main__":
    main()

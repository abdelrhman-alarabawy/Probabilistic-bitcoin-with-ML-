import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# === CONFIG ===
SEED = 42
CSV_PATH = Path(
    r"D:\GitHub\bitcoin-probabilistic-learning\data\processed\features_1h_ALL-2025_merged_prev_indicators_labeled.csv"
)
OUTPUT_DIR = Path(r"D:\GitHub\bitcoin-probabilistic-learning\bn_outputs")

TIMESTAMP_COL_HINT = "nts-utc"
LABEL_COL_HINT = "Candle_type"
TRAIN_FRACTION = 0.7

STATE_WINDOW = 24
TREND_FAST = 6
TREND_SLOW = 24
STATE_BINS = 3
INDICATOR_BINS = 3

MAX_PARENTS = 3
MAX_INDICATORS_FOR_STRUCTURE = 25
MAX_ABLATION_INDICATORS = 10
MAX_ABLATION_ROWS = 50
HC_MAX_ITER = 800
STRUCTURE_MAX_ROWS = 5000
PC_MAX_ROWS = 4000
MIN_CLASS_FRAC = 0.01


def normalize_key(name: str) -> str:
    return "".join(ch for ch in name.lower().strip() if ch.isalnum())


def strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def find_column(columns: Iterable[str], hint: str, fallbacks: Iterable[str]) -> str:
    norm_map = {normalize_key(c): c for c in columns}
    targets = [normalize_key(hint)]
    if targets[0].startswith("n"):
        targets.append(targets[0][1:])
    targets.extend([normalize_key(f) for f in fallbacks])
    for t in targets:
        if t in norm_map:
            return norm_map[t]
    for c in columns:
        n = normalize_key(c)
        if "ts" in n and "utc" in n:
            return c
    raise ValueError(f"Could not locate column for hint '{hint}'. Columns: {list(columns)}")


def detect_ohlcv(columns: Iterable[str]) -> Dict[str, str]:
    norm_map = {normalize_key(c): c for c in columns}
    result = {}
    for key in ["open", "high", "low", "close", "volume"]:
        if normalize_key(key) in norm_map:
            result[key] = norm_map[normalize_key(key)]
    return result


def drop_leakage_columns(columns: Iterable[str]) -> List[str]:
    blocked = ("t+1", "future", "next")
    keep = []
    for col in columns:
        col_l = col.lower()
        if any(b in col_l for b in blocked):
            continue
        keep.append(col)
    return keep


def make_quantile_edges(values: pd.Series, bins: int) -> Optional[List[float]]:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if clean.nunique() <= 1:
        return None
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(clean, quantiles))
    if len(edges) < 3:
        return None
    return edges.tolist()


def discretize_series(values: pd.Series, edges: List[float]) -> pd.Series:
    labels = [f"bin_{i}" for i in range(len(edges) - 1)]
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.cut(numeric, bins=edges, labels=labels, include_lowest=True)


def mutual_information(x: pd.Series, y: pd.Series) -> float:
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if df.empty:
        return 0.0
    joint = pd.crosstab(df["x"], df["y"], normalize=True)
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    mi = 0.0
    for i in joint.index:
        for j in joint.columns:
            pxy = joint.loc[i, j]
            if pxy <= 0:
                continue
            mi += pxy * math.log(pxy / (px[i] * py[j] + 1e-12) + 1e-12)
    return float(mi)


def conditional_mutual_information(
    x: pd.Series, y: pd.Series, cond_df: pd.DataFrame
) -> float:
    df = pd.DataFrame({"x": x, "y": y}).join(cond_df).dropna()
    if df.empty:
        return 0.0
    total = len(df)
    cmi = 0.0
    for _, group in df.groupby(list(cond_df.columns)):
        weight = len(group) / total
        if len(group) < 2:
            continue
        cmi += weight * mutual_information(group["x"], group["y"])
    return float(cmi)


def macro_f1_score(y_true: List[str], y_pred: List[str]) -> float:
    labels = sorted(set(y_true))
    if not labels:
        return 0.0
    f1s = []
    for label in labels:
        tp = sum((yt == label) and (yp == label) for yt, yp in zip(y_true, y_pred))
        fp = sum((yt != label) and (yp == label) for yt, yp in zip(y_true, y_pred))
        fn = sum((yt == label) and (yp != label) for yt, yp in zip(y_true, y_pred))
        if tp == 0:
            f1s.append(0.0)
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if precision + recall == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * precision * recall / (precision + recall))
    return float(sum(f1s) / len(f1s))


def prepare_states(df: pd.DataFrame, ohlcv: Dict[str, str]) -> pd.DataFrame:
    df = df.copy()
    close = df[ohlcv["close"]]
    df["_return"] = np.log(close).diff()
    df["_volatility"] = df["_return"].rolling(STATE_WINDOW).std()
    ma_fast = close.rolling(TREND_FAST).mean()
    ma_slow = close.rolling(TREND_SLOW).mean()
    df["_trend"] = (ma_fast - ma_slow) / ma_slow
    prev_close = close.shift(1)
    tr_components = pd.concat(
        [
            (df[ohlcv["high"]] - df[ohlcv["low"]]).abs(),
            (df[ohlcv["high"]] - prev_close).abs(),
            (df[ohlcv["low"]] - prev_close).abs(),
        ],
        axis=1,
    )
    df["_true_range"] = tr_components.max(axis=1)
    df["_range"] = df["_true_range"].rolling(STATE_WINDOW).mean()
    df["_volume_state"] = df[ohlcv["volume"]].rolling(STATE_WINDOW).mean()
    return df


def build_allowed_edges(
    states: List[str], indicators: List[str], label: str
) -> List[Tuple[str, str]]:
    allowed = []
    for s in states:
        allowed.append((s, label))
        for ind in indicators:
            allowed.append((s, ind))
    for ind in indicators:
        allowed.append((ind, label))
    return allowed


def cpd_prob(cpd, y_val: str, parent_vals: Dict[str, str]) -> float:
    state_names = cpd.state_names
    if y_val not in state_names[cpd.variable]:
        return 1e-12
    idx = [state_names[cpd.variable].index(y_val)]
    for parent in cpd.variables[1:]:
        states = state_names[parent]
        val = parent_vals.get(parent)
        if val not in states:
            return 1e-12
        idx.append(states.index(val))
    return float(cpd.values[tuple(idx)])


def log_likelihood_y(model, data: pd.DataFrame, label: str) -> float:
    cpd = model.get_cpds(label)
    parents = list(model.get_parents(label))
    total = 0.0
    count = 0
    for _, row in data.iterrows():
        y_val = row[label]
        parent_vals = {p: row[p] for p in parents}
        prob = cpd_prob(cpd, y_val, parent_vals)
        total += math.log(prob + 1e-12)
        count += 1
    return total / max(count, 1)


def best_model_predict(model, data: pd.DataFrame, label: str) -> List[str]:
    cpd = model.get_cpds(label)
    parents = list(model.get_parents(label))
    state_names = cpd.state_names[label]
    preds = []
    for _, row in data.iterrows():
        parent_vals = {p: row[p] for p in parents}
        probs = [cpd_prob(cpd, y_state, parent_vals) for y_state in state_names]
        preds.append(state_names[int(np.argmax(probs))])
    return preds


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    df = strip_columns(df)
    df = df[drop_leakage_columns(df.columns)]

    ts_col = find_column(df.columns, TIMESTAMP_COL_HINT, ["ts_utc", "timestamp"])
    label_col = find_column(df.columns, LABEL_COL_HINT, ["candle_type", "label"])

    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=[ts_col])
    df = df.sort_values(ts_col).reset_index(drop=True)

    ohlcv_map = detect_ohlcv(df.columns)
    if len(ohlcv_map) < 5:
        missing = {k for k in ["open", "high", "low", "close", "volume"] if k not in ohlcv_map}
        raise ValueError(f"Missing OHLCV columns: {missing}")

    for col in ohlcv_map.values():
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[label_col] = df[label_col].astype(str)

    df = prepare_states(df, ohlcv_map)

    split_idx = int(len(df) * TRAIN_FRACTION)
    split_ts = df.loc[split_idx, ts_col] if split_idx < len(df) else df[ts_col].iloc[-1]

    train_mask = df[ts_col] <= split_ts
    train_df = df.loc[train_mask].copy()
    test_df = df.loc[~train_mask].copy()

    # --- Build state RVs ---
    state_raw_cols = {
        "VolState": "_volatility",
        "TrendState": "_trend",
        "RangeState": "_range",
        "VolumeState": "_volume_state",
    }
    state_bins = {}
    for state, raw_col in state_raw_cols.items():
        edges = make_quantile_edges(train_df[raw_col], STATE_BINS)
        if edges is None:
            raise ValueError(f"Cannot bin state {state} due to insufficient variation.")
        state_bins[state] = edges
        df[state] = discretize_series(df[raw_col], edges)

    # --- Build indicator RVs ---
    exclude = set(ohlcv_map.values()) | {ts_col, label_col} | set(state_raw_cols.values())
    numeric_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if col in state_raw_cols:
            continue
        if col in state_raw_cols.values():
            continue
        if df[col].dtype == "O":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)

    indicator_cols = []
    for col in numeric_cols:
        if df[col].nunique(dropna=True) < 5:
            continue
        indicator_cols.append(col)

    indicator_bins = {}
    for col in indicator_cols:
        edges = make_quantile_edges(train_df[col], INDICATOR_BINS)
        if edges is None:
            continue
        indicator_bins[col] = edges
        df[f"{col}_bin"] = discretize_series(df[col], edges)

    indicator_bin_cols = [f"{col}_bin" for col in indicator_bins.keys()]
    state_cols = list(state_raw_cols.keys())

    df_model = df[[ts_col, label_col] + state_cols + indicator_bin_cols].copy()
    df_model = df_model.dropna()

    train_df = df_model[df_model[ts_col] <= split_ts].copy()
    test_df = df_model[df_model[ts_col] > split_ts].copy()

    # Merge rare classes if needed
    train_counts = train_df[label_col].value_counts(normalize=True)
    rare_classes = train_counts[train_counts < MIN_CLASS_FRAC].index.tolist()
    if rare_classes:
        train_df[label_col] = train_df[label_col].replace(rare_classes, "Other")
        test_df[label_col] = test_df[label_col].replace(rare_classes, "Other")

    # Convert to string for discrete BN
    for col in state_cols + indicator_bin_cols + [label_col]:
        train_df[col] = train_df[col].astype(str)
        test_df[col] = test_df[col].astype(str)

    # --- Prefilter indicators for structure learning ---
    mi_scores = []
    for col in indicator_bin_cols:
        mi_scores.append((col, mutual_information(train_df[col], train_df[label_col])))
    mi_scores.sort(key=lambda x: x[1], reverse=True)
    top_indicator_bins = [c for c, _ in mi_scores[:MAX_INDICATORS_FOR_STRUCTURE]]

    structure_cols = state_cols + top_indicator_bins + [label_col]

    try:
        from pgmpy.estimators import (
            BayesianEstimator,
            BicScore,
            BDeuScore,
            HillClimbSearch,
            PC,
        )
        from pgmpy.inference import VariableElimination
        from pgmpy.models import BayesianNetwork
    except Exception as exc:
        raise RuntimeError(
            "pgmpy is required to run this pipeline. Install pgmpy and retry."
        ) from exc

    allowed_edges = build_allowed_edges(state_cols, top_indicator_bins, label_col)

    # --- Method 1: Score-based (Hill Climb + BIC) ---
    structure_df = train_df[structure_cols].copy()
    if len(structure_df) > STRUCTURE_MAX_ROWS:
        structure_df = structure_df.iloc[:STRUCTURE_MAX_ROWS].copy()

    hc = HillClimbSearch(structure_df)
    bic = BicScore(train_df[structure_cols])
    model_score = hc.estimate(
        scoring_method=bic,
        white_list=allowed_edges,
        max_indegree=MAX_PARENTS,
        max_iter=HC_MAX_ITER,
        show_progress=False,
    )
    score_model = BayesianNetwork(model_score.edges())
    score_model.add_nodes_from(structure_cols)
    score_model.fit(train_df[structure_cols], estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=5)

    # --- Method 2: Constraint-based (PC) ---
    pc_df = train_df[structure_cols].copy()
    if len(pc_df) > PC_MAX_ROWS:
        pc_df = pc_df.iloc[:PC_MAX_ROWS].copy()

    pc = PC(pc_df)
    pc_graph = pc.estimate(
        ci_test="chi_square",
        max_cond_vars=1,
        significance_level=0.05,
        variant="stable",
    )
    pc_edges = []
    for u, v in pc_graph.edges():
        if (u, v) in allowed_edges:
            pc_edges.append((u, v))
        elif (v, u) in allowed_edges:
            pc_edges.append((v, u))

    pc_model = BayesianNetwork(pc_edges)
    pc_model.add_nodes_from(structure_cols)
    pc_model.fit(train_df[structure_cols], estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=5)

    # --- Evaluate models ---
    score_ll = log_likelihood_y(score_model, test_df[structure_cols], label_col)
    pc_ll = log_likelihood_y(pc_model, test_df[structure_cols], label_col)

    score_preds = best_model_predict(score_model, test_df[structure_cols], label_col)
    pc_preds = best_model_predict(pc_model, test_df[structure_cols], label_col)
    y_true = test_df[label_col].tolist()

    score_acc = float(np.mean([yt == yp for yt, yp in zip(y_true, score_preds)]))
    pc_acc = float(np.mean([yt == yp for yt, yp in zip(y_true, pc_preds)]))
    score_f1 = macro_f1_score(y_true, score_preds)
    pc_f1 = macro_f1_score(y_true, pc_preds)

    if score_ll >= pc_ll:
        best_model = score_model
        best_method = "score_based"
        best_ll = score_ll
        best_acc = score_acc
        best_f1 = score_f1
    else:
        best_model = pc_model
        best_method = "constraint_based"
        best_ll = pc_ll
        best_acc = pc_acc
        best_f1 = pc_f1

    # --- Output directory ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- bn_edges.csv ---
    edge_rows = []
    for node in score_model.nodes():
        local_score = bic.local_score(node, score_model.get_parents(node))
        for parent in score_model.get_parents(node):
            edge_rows.append(
                {"source": parent, "target": node, "method": "score_based", "score": local_score}
            )
    for u, v in pc_model.edges():
        edge_rows.append({"source": u, "target": v, "method": "constraint_based", "score": "p=na"})

    edges_path = OUTPUT_DIR / "bn_edges.csv"
    pd.DataFrame(edge_rows).to_csv(edges_path, index=False)

    # --- bn_graph.png ---
    try:
        import matplotlib.pyplot as plt
        import networkx as nx

        graph = nx.DiGraph()
        graph.add_nodes_from(best_model.nodes())
        graph.add_edges_from(best_model.edges())
        pos = nx.spring_layout(graph, seed=SEED, k=0.6)
        plt.figure(figsize=(14, 10))
        nx.draw_networkx_nodes(graph, pos, node_size=900, node_color="#E5E7EB", edgecolors="#111827")
        nx.draw_networkx_edges(graph, pos, arrows=True, arrowstyle="->", arrowsize=12, width=1.2)
        nx.draw_networkx_labels(graph, pos, font_size=8)
        plt.axis("off")
        graph_path = OUTPUT_DIR / "bn_graph.png"
        plt.tight_layout()
        plt.savefig(graph_path, dpi=200)
        plt.close()
    except Exception as exc:
        raise RuntimeError("Failed to generate bn_graph.png. Ensure matplotlib/networkx installed.") from exc

    # --- Save BN model ---
    model_path = OUTPUT_DIR / "bn_model.pkl"
    try:
        import joblib

        joblib.dump(best_model, model_path)
    except Exception:
        import pickle

        with model_path.open("wb") as handle:
            pickle.dump(best_model, handle)

    # --- Indicator usefulness metrics ---
    test_states = test_df[state_cols]
    usefulness_rows = []
    best_graph = best_model

    import networkx as nx

    undirected = nx.Graph()
    undirected.add_nodes_from(best_graph.nodes())
    undirected.add_edges_from(best_graph.edges())

    mb = set(best_graph.get_markov_blanket(label_col))
    parents_y = set(best_graph.get_parents(label_col))

    # Baseline log-likelihood using full evidence on a sample
    inference = VariableElimination(best_graph)
    test_sample = test_df.copy()
    if len(test_sample) > MAX_ABLATION_ROWS:
        test_sample = test_sample.iloc[:MAX_ABLATION_ROWS].copy()

    def loglik_infer(data: pd.DataFrame, evidence_cols: List[str]) -> float:
        total = 0.0
        for _, row in data.iterrows():
            evidence = {col: row[col] for col in evidence_cols}
            q = inference.query([label_col], evidence=evidence, show_progress=False)
            y_val = row[label_col]
            prob = float(q.values[q.state_names[label_col].index(y_val)])
            total += math.log(prob + 1e-12)
        return total / max(len(data), 1)

    indicator_in_model = [c for c in indicator_bin_cols if c in best_graph.nodes()]
    ablation_targets = indicator_in_model
    if len(ablation_targets) > MAX_ABLATION_INDICATORS:
        ablation_targets = [c for c, _ in mi_scores[:MAX_ABLATION_INDICATORS]]
    evidence_cols_full = state_cols + ablation_targets
    baseline_ll = loglik_infer(test_sample, evidence_cols_full)

    ablation_map = {}
    for col in ablation_targets:
        evidence_cols = [c for c in evidence_cols_full if c != col]
        ablation_ll = loglik_infer(test_sample, evidence_cols)
        ablation_map[col] = baseline_ll - ablation_ll

    for col in indicator_bin_cols:
        mi = mutual_information(test_df[col], test_df[label_col])
        cmi = conditional_mutual_information(test_df[col], test_df[label_col], test_states)
        in_mb = col in mb
        is_parent = col in parents_y
        try:
            distance = nx.shortest_path_length(undirected, source=col, target=label_col)
        except Exception:
            distance = None

        usefulness_rows.append(
            {
                "indicator": col,
                "mi": mi,
                "cmi": cmi,
                "is_parent_of_y": is_parent,
                "in_markov_blanket": in_mb,
                "distance_to_y": distance if distance is not None else "",
                "ablation_ll_drop": ablation_map.get(col, ""),
            }
        )

    usefulness_df = pd.DataFrame(usefulness_rows)

    def normalize_series(values: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
        if numeric.max() == numeric.min():
            return pd.Series([0.0] * len(numeric), index=values.index)
        return (numeric - numeric.min()) / (numeric.max() - numeric.min())

    graph_score = (
        usefulness_df["is_parent_of_y"].astype(int)
        + 0.5 * usefulness_df["in_markov_blanket"].astype(int)
    )
    distance_score = 1 / (pd.to_numeric(usefulness_df["distance_to_y"], errors="coerce") + 1)
    usefulness_df["graph_score"] = graph_score + distance_score.fillna(0.0)

    usefulness_df["mi_norm"] = normalize_series(usefulness_df["mi"])
    usefulness_df["cmi_norm"] = normalize_series(usefulness_df["cmi"])
    usefulness_df["graph_norm"] = normalize_series(usefulness_df["graph_score"])
    usefulness_df["ablation_norm"] = normalize_series(usefulness_df["ablation_ll_drop"])

    usefulness_df["combined_score"] = (
        0.35 * usefulness_df["mi_norm"]
        + 0.25 * usefulness_df["cmi_norm"]
        + 0.2 * usefulness_df["graph_norm"]
        + 0.2 * usefulness_df["ablation_norm"]
    )
    usefulness_df = usefulness_df.sort_values("combined_score", ascending=False).reset_index(drop=True)
    top_k = max(20, int(0.1 * len(usefulness_df)))
    usefulness_df["rank"] = usefulness_df.index + 1
    usefulness_df["useful"] = usefulness_df["rank"] <= top_k

    usefulness_path = OUTPUT_DIR / "indicator_usefulness.csv"
    usefulness_df.to_csv(usefulness_path, index=False)

    # --- metadata.json ---
    rv_list = []
    rv_list.append({"name": label_col, "type": "target", "domain": "categorical"})
    for state in state_cols:
        rv_list.append(
            {
                "name": state,
                "type": "state",
                "bins": state_bins[state],
            }
        )
    for col in indicator_bins:
        rv_list.append(
            {
                "name": f"{col}_bin",
                "type": "indicator",
                "bins": indicator_bins[col],
            }
        )

    metadata = {
        "dataset_path": str(CSV_PATH),
        "timestamp_col": ts_col,
        "label_col": label_col,
        "train_fraction": TRAIN_FRACTION,
        "train_end_timestamp": str(split_ts),
        "test_start_timestamp": str(split_ts),
        "rvs": rv_list,
        "leakage_checks": {
            "removed_columns_with": ["t+1", "future", "next"],
            "label_excluded_from_features": True,
        },
        "rare_classes_merged": rare_classes,
        "indicator_bins_count": len(indicator_bins),
        "structure_indicators_used": len(top_indicator_bins),
        "ablation_indicators_covered": len(ablation_targets),
        "ablation_rows": len(test_sample),
    }
    meta_path = OUTPUT_DIR / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))

    # --- variables_report.md ---
    report_lines = [
        "# Random Variables Report",
        "",
        "## Target RV",
        f"- {label_col}: categorical label for candle type.",
        "",
        "## Market State RVs",
    ]
    for state, raw in state_raw_cols.items():
        report_lines.append(
            f"- {state}: discretized from {raw} using {STATE_BINS} quantile bins."
        )
    report_lines.append("")
    report_lines.append("## Indicator RVs")
    report_lines.append(
        f"- {len(indicator_bins)} indicators discretized into {INDICATOR_BINS} quantile bins (train-only)."
    )
    report_path = OUTPUT_DIR / "variables_report.md"
    report_path.write_text("\n".join(report_lines))

    # --- evaluation.md ---
    eval_lines = [
        "# BN Evaluation",
        "",
        f"Best method: {best_method}",
        "",
        "## Score-based (Hill Climb + BIC)",
        f"- test log-likelihood (Y): {score_ll:.6f}",
        f"- accuracy: {score_acc:.4f}",
        f"- macro-F1: {score_f1:.4f}",
        "",
        "## Constraint-based (PC)",
        f"- test log-likelihood (Y): {pc_ll:.6f}",
        f"- accuracy: {pc_acc:.4f}",
        f"- macro-F1: {pc_f1:.4f}",
        "",
        "## Interpretation",
        "- Higher log-likelihood indicates better calibrated P(Y|evidence).",
    ]
    eval_path = OUTPUT_DIR / "evaluation.md"
    eval_path.write_text("\n".join(eval_lines))

    # --- Example inference cases ---
    example_lines = [
        "",
        "## Example Inference Cases",
    ]
    top_indicator_examples = [c for c in usefulness_df.head(5)["indicator"].tolist() if c in best_graph.nodes()]
    example_rows = test_df.sample(n=min(5, len(test_df)), random_state=SEED)
    inference_engine = VariableElimination(best_model)
    for idx, (_, row) in enumerate(example_rows.iterrows(), start=1):
        evidence = {state: row[state] for state in state_cols}
        for ind in top_indicator_examples:
            evidence[ind] = row[ind]
        posterior = inference_engine.query([label_col], evidence=evidence, show_progress=False)
        example_lines.append(f"- Case {idx}: evidence={evidence} -> P(Y)={posterior.values}")

    eval_path.write_text(eval_path.read_text() + "\n" + "\n".join(example_lines))

    # --- Print top indicators ---
    print("\nTop 20 indicators:")
    for _, row in usefulness_df.head(20).iterrows():
        reason = []
        if row["is_parent_of_y"]:
            reason.append("parent of Y")
        if row["in_markov_blanket"]:
            reason.append("in Markov blanket")
        if pd.to_numeric(row["cmi"], errors="coerce") > usefulness_df["cmi"].median():
            reason.append("high CMI")
        if pd.to_numeric(row["ablation_ll_drop"], errors="coerce") > 0:
            reason.append("positive ablation gain")
        note = "; ".join(reason) if reason else "weak conditional signal"
        print(f"- {row['indicator']}: {note}")

    print(f"\nBN graph saved to: {graph_path}")
    print(f"BN model saved to: {model_path}")
    print(f"Indicator usefulness saved to: {usefulness_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)

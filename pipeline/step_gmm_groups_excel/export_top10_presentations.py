#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


PREFERRED_TIE_ORDER = ["long", "short", "skip"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export top-10 model presentations to Excel.")
    parser.add_argument(
        "--labeled_root",
        type=str,
        default="pipeline/step_gmm_groups_labeling/results",
        help="Root folder containing group/model_rankXX/labeled.csv.",
    )
    parser.add_argument(
        "--out_xlsx",
        type=str,
        default="pipeline/step_gmm_groups_excel/results/top10_presentations.xlsx",
        help="Output Excel file path.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler()],
    )


def find_groups(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.lower() != "all"])


def load_gmm_config(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def infer_k_from_columns(df: pd.DataFrame) -> Optional[int]:
    prob_cols = [c for c in df.columns if c.startswith("gmm_prob_state_")]
    if prob_cols:
        return len(prob_cols)
    if "gmm_hard_state" in df.columns:
        try:
            return int(df["gmm_hard_state"].max()) + 1
        except Exception:
            return None
    return None


def dominant_label(series: pd.Series) -> str:
    if series.empty:
        return "skip"
    counts = series.value_counts()
    max_count = counts.max()
    candidates = [label for label, count in counts.items() if count == max_count]
    for label in PREFERRED_TIE_ORDER:
        if label in candidates:
            return label
    return sorted(candidates)[0]


def build_state_maps(df: pd.DataFrame, k: Optional[int]) -> Tuple[str, str, str]:
    counts = (
        df.groupby(["gmm_hard_state", "trade_label"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=PREFERRED_TIE_ORDER, fill_value=0)
    )
    if k is None:
        k = int(counts.index.max()) + 1 if not counts.empty else 0
    presented = []
    counts_parts = []
    pct_parts = []

    for state in range(0, int(k)):
        if state not in counts.index:
            state_counts = pd.Series({label: 0 for label in PREFERRED_TIE_ORDER})
        else:
            state_counts = counts.loc[state]
        total = int(state_counts.sum())
        expanded = []
        for label in PREFERRED_TIE_ORDER:
            expanded.extend([label] * int(state_counts[label]))
        dom = dominant_label(pd.Series(expanded))
        presented.append(f"state{state}={dom}")
        counts_parts.append(
            f"state{state}={{long:{int(state_counts['long'])}, short:{int(state_counts['short'])}, skip:{int(state_counts['skip'])}}}"
        )
        pct_parts.append(
            f"state{state}={{long:{(state_counts['long'] / total * 100) if total else 0:.1f}%, "
            f"short:{(state_counts['short'] / total * 100) if total else 0:.1f}%, "
            f"skip:{(state_counts['skip'] / total * 100) if total else 0:.1f}%}}"
        )

    return ", ".join(presented), "; ".join(counts_parts), "; ".join(pct_parts)


def compute_overall_label_stats(df: pd.DataFrame) -> Tuple[str, str]:
    counts = df["trade_label"].value_counts().reindex(PREFERRED_TIE_ORDER, fill_value=0)
    total = int(counts.sum())
    counts_str = f"long:{int(counts['long'])}, short:{int(counts['short'])}, skip:{int(counts['skip'])}"
    pcts_str = (
        f"long:{(counts['long'] / total * 100) if total else 0:.1f}%, "
        f"short:{(counts['short'] / total * 100) if total else 0:.1f}%, "
        f"skip:{(counts['skip'] / total * 100) if total else 0:.1f}%"
    )
    return counts_str, pcts_str


def process_model(model_dir: Path) -> Optional[Dict[str, object]]:
    labeled_path = model_dir / "labeled.csv"
    if not labeled_path.exists():
        logging.warning("Missing labeled.csv: %s", labeled_path)
        return None
    df = pd.read_csv(labeled_path)
    if "gmm_hard_state" not in df.columns or "trade_label" not in df.columns:
        logging.warning("Missing required columns in %s", labeled_path)
        return None

    df["trade_label"] = df["trade_label"].astype(str).str.strip().str.lower()
    df = df[df["trade_label"].isin(PREFERRED_TIE_ORDER)].copy()
    if df.empty:
        logging.warning("No valid trade labels in %s", labeled_path)
        return None

    gmm_cfg = load_gmm_config(model_dir / "gmm_config.json")
    k = gmm_cfg.get("n_components")
    if k is None:
        k = infer_k_from_columns(df)
    k = int(k) if k is not None else None

    cov = gmm_cfg.get("covariance_type", "")
    features = gmm_cfg.get("selected_features_used", gmm_cfg.get("feature_columns_used", ""))
    if isinstance(features, list):
        features = "|".join([str(x) for x in features])

    presented_map, counts_str, pct_str = build_state_maps(df, k=k)
    overall_counts, overall_pcts = compute_overall_label_stats(df)

    return {
        "K": k,
        "covariance_type": cov,
        "feature_set": features,
        "presented_state_map": presented_map,
        "state_label_counts": counts_str,
        "state_label_pcts": pct_str,
        "overall_label_counts": overall_counts,
        "overall_label_pcts": overall_pcts,
    }


def ensure_rank_rows(group_name: str, group_dir: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for rank in range(1, 11):
        model_dir = group_dir / f"model_rank{rank:02d}"
        base = {"group_name": group_name, "model_rank": rank}
        result = process_model(model_dir)
        if result is None:
            rows.append({**base})
        else:
            rows.append({**base, **result})
    return rows


def auto_size_and_wrap(workbook, sheet_name: str, wrap_cols: List[int]) -> None:
    ws = workbook[sheet_name]
    ws.freeze_panes = "A2"
    for cell in ws[1]:
        cell.font = cell.font.copy(bold=True)

    for col_idx in wrap_cols:
        for row in ws.iter_rows(min_row=1, min_col=col_idx, max_col=col_idx):
            for cell in row:
                cell.alignment = cell.alignment.copy(wrap_text=True, vertical="top")

    for col in ws.columns:
        max_len = 0
        col_letter = col[0].column_letter
        for cell in col:
            if cell.value is None:
                continue
            max_len = max(max_len, len(str(cell.value)))
        ws.column_dimensions[col_letter].width = min(max_len + 2, 80)


def main() -> None:
    args = parse_args()
    setup_logging()
    labeled_root = Path(args.labeled_root)
    out_path = Path(args.out_xlsx)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not labeled_root.exists():
        raise FileNotFoundError(f"Labeled root not found: {labeled_root}")

    group_dirs = find_groups(labeled_root)
    if not group_dirs:
        raise FileNotFoundError(f"No group directories found in {labeled_root}")

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for group_dir in group_dirs:
            group_name = group_dir.name
            sheet_name = group_name[:31]
            rows = ensure_rank_rows(group_name, group_dir)
            df = pd.DataFrame(
                rows,
                columns=[
                    "group_name",
                    "model_rank",
                    "K",
                    "covariance_type",
                    "feature_set",
                    "presented_state_map",
                    "state_label_counts",
                    "state_label_pcts",
                    "overall_label_counts",
                    "overall_label_pcts",
                ],
            )
            df.to_excel(writer, sheet_name=sheet_name, index=False)

        workbook = writer.book
        for group_dir in group_dirs:
            sheet_name = group_dir.name[:31]
            auto_size_and_wrap(workbook, sheet_name, wrap_cols=[6, 7, 8])

    logging.info("Wrote Excel: %s", out_path)


if __name__ == "__main__":
    main()

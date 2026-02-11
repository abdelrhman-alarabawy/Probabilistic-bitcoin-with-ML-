#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

from src.gating import apply_group_gate, combine_gates
from src.load_align import align_groups, discover_groups, load_group_labeled
from src.reports import compute_label_agreement, compute_pass_rates, summarize_final_labels
from src.thresholds import compute_thresholds
from src.voting import majority_vote, reference_label, weighted_vote


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-group gating over labeled GMM outputs.")
    parser.add_argument("--config", type=str, default="pipeline/step_gmm_multi_group_gating/config.yaml")
    parser.add_argument("--labeled_root", type=str, help="Root folder of labeled outputs.")
    parser.add_argument("--model_rank", type=int, help="Model rank to use per group (1..10).")
    parser.add_argument("--mode", type=str, choices=["strict_and", "k_of_n", "weighted_score"])
    parser.add_argument("--k", type=int, help="K for k-of-n mode.")
    parser.add_argument("--align", type=str, choices=["intersection", "union"])
    parser.add_argument("--prob_q", type=float)
    parser.add_argument("--ent_q", type=float)
    parser.add_argument("--rare_q", type=float)
    parser.add_argument("--train_frac", type=float)
    parser.add_argument("--vote", type=str, choices=["majority", "reference", "weighted"])
    parser.add_argument("--out_dir", type=str, help="Output directory.")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config must be a dictionary.")
    return data


def apply_overrides(cfg: Dict[str, object], args: argparse.Namespace) -> Dict[str, object]:
    merged = dict(cfg)
    if args.labeled_root:
        merged["labeled_root"] = args.labeled_root
    if args.model_rank:
        merged["model_rank"] = int(args.model_rank)
    if args.mode:
        merged["mode"] = args.mode
    if args.k is not None:
        merged["k_required"] = int(args.k)
    if args.align:
        merged["align_mode"] = args.align
    if args.prob_q is not None:
        merged["prob_q"] = float(args.prob_q)
    if args.ent_q is not None:
        merged["ent_q"] = float(args.ent_q)
    if args.rare_q is not None:
        merged["rare_q"] = float(args.rare_q)
    if args.train_frac is not None:
        merged["train_fraction"] = float(args.train_frac)
    if args.vote:
        merged["vote_mode"] = args.vote
    if args.out_dir:
        merged["out_dir"] = args.out_dir
    return merged


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler()],
    )


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_config(Path(args.config)), args)
    setup_logging()

    labeled_root = Path(cfg.get("labeled_root", "pipeline/step_gmm_groups_labeling/results"))
    model_rank = int(cfg.get("model_rank", 1))
    align_mode = str(cfg.get("align_mode", "intersection"))
    mode = str(cfg.get("mode", "k_of_n"))
    k_required = int(cfg.get("k_required", 4))
    prob_q = float(cfg.get("prob_q", 0.90))
    ent_q = float(cfg.get("ent_q", 0.10))
    rare_q = float(cfg.get("rare_q", 0.15))
    train_fraction = float(cfg.get("train_fraction", 0.70))
    vote_mode = str(cfg.get("vote_mode", "majority"))
    out_dir = Path(cfg.get("out_dir", "pipeline/step_gmm_multi_group_gating/results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    group_dirs = discover_groups(labeled_root)
    if not group_dirs:
        raise FileNotFoundError(f"No group directories found in {labeled_root}")

    group_frames: Dict[str, pd.DataFrame] = {}
    for group_dir in group_dirs:
        name, df = load_group_labeled(group_dir, model_rank=model_rank)
        group_frames[name] = df

    aligned = align_groups(group_frames, align_mode=align_mode)
    group_names = list(group_frames.keys())

    thresholds_out: Dict[str, object] = {}
    for g in group_names:
        df_g = aligned[[f"{g}_hard_state"]].copy()
        if f"{g}_probmax" in aligned.columns:
            df_g[f"{g}_probmax"] = aligned[f"{g}_probmax"]
        if f"{g}_entropy" in aligned.columns:
            df_g[f"{g}_entropy"] = aligned[f"{g}_entropy"]

        thresholds = compute_thresholds(
            df=df_g.rename(
                columns={
                    f"{g}_probmax": "prob",
                    f"{g}_entropy": "ent",
                    f"{g}_hard_state": "state",
                }
            ),
            prob_col="prob",
            ent_col="ent",
            state_col="state",
            train_fraction=train_fraction,
            prob_q=prob_q,
            ent_q=ent_q,
            rare_q=rare_q,
        )

        thresholds_out[g] = {
            "probmax_threshold": thresholds.probmax_thr,
            "entropy_threshold": thresholds.entropy_thr,
            "rarity_threshold": thresholds.rarity_thr,
            "state_freq": thresholds.state_freq,
            "train_rows": thresholds.train_rows,
        }

        gate_pass, gate_score = apply_group_gate(aligned, g, thresholds)
        aligned[f"{g}_gate_pass"] = gate_pass
        aligned[f"{g}_gate_score"] = gate_score

    weights = cfg.get("weights", {})
    final_pass, final_score = combine_gates(
        df=aligned,
        group_names=group_names,
        mode=mode,
        k_required=k_required,
        weights=weights,
        score_threshold=float(cfg.get("score_threshold", 0.6)),
    )
    aligned["final_pass"] = final_pass
    if mode == "weighted_score":
        aligned["final_score"] = final_score
    else:
        aligned["final_score"] = pd.NA

    if vote_mode == "reference":
        ref_group = str(cfg.get("reference_group", group_names[0]))
        aligned["final_label"] = aligned.apply(lambda r: reference_label(r, ref_group), axis=1)
    elif vote_mode == "weighted":
        weights_map = {g: float(weights.get(g, 1.0)) for g in group_names}
        aligned["final_label"] = aligned.apply(lambda r: weighted_vote(r, group_names, weights_map), axis=1)
    else:
        aligned["final_label"] = aligned.apply(lambda r: majority_vote(r, group_names), axis=1)

    combined_path = out_dir / "combined_time_aligned.csv"
    aligned.to_csv(combined_path, index=False)

    passed = aligned[aligned["final_pass"] == 1].copy()
    final_path = out_dir / "final_pass_candles.csv"
    passed.to_csv(final_path, index=False)

    summary = {
        "mode": mode,
        "k_required": k_required,
        "align_mode": align_mode,
        "prob_q": prob_q,
        "ent_q": ent_q,
        "rare_q": rare_q,
        "train_fraction": train_fraction,
        "vote_mode": vote_mode,
        "n_total": int(aligned.shape[0]),
        "n_pass": int(passed.shape[0]),
        "coverage_pct": float(passed.shape[0] / aligned.shape[0]) if aligned.shape[0] else 0.0,
    }
    summary.update(summarize_final_labels(passed))
    pred_counts = {
        "long": int((passed["final_label"] == "long").sum()),
        "short": int((passed["final_label"] == "short").sum()),
        "skip": int((passed["final_label"] == "skip").sum()),
    }
    denom = pred_counts["long"] + pred_counts["short"] + pred_counts["skip"]
    coverage_trade = (pred_counts["long"] + pred_counts["short"]) / denom if denom else 0.0
    summary["pred_counts"] = pred_counts
    summary["coverage_trade"] = coverage_trade
    summary["final_pass_count"] = int(passed.shape[0])
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    with (out_dir / "per_group_thresholds.json").open("w", encoding="utf-8") as f:
        json.dump(thresholds_out, f, indent=2, sort_keys=True)

    pass_rates = compute_pass_rates(aligned, group_names)
    pass_rates.to_csv(out_dir / "per_group_pass_rates.csv", index=False)
    gate_pass_rate_per_group = {row["group_name"]: float(row["pass_rate"]) for _, row in pass_rates.iterrows()}
    summary["gate_pass_rate_per_group"] = gate_pass_rate_per_group

    agreement = compute_label_agreement(aligned, group_names)
    agreement.to_csv(out_dir / "label_vote_agreement.csv", index=False)

    logging.info(
        "Done. n_total=%d n_pass=%d coverage=%.3f final_long=%.3f final_short=%.3f final_skip=%.3f",
        summary["n_total"],
        summary["n_pass"],
        summary["coverage_pct"],
        summary["final_long_pct"],
        summary["final_short_pct"],
        summary["final_skip_pct"],
    )
    logging.info("pred_counts=%s", pred_counts)
    logging.info("coverage_trade=%.4f", coverage_trade)
    logging.info("gate_pass_rate_per_group=%s", gate_pass_rate_per_group)
    logging.info("final_pass_count=%d", summary["final_pass_count"])


if __name__ == "__main__":
    main()

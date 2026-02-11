from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import load_config
from run import _preprocess
from src.data import load_data
from src.eval_state_gmms import state_component_posteriors
from src.features import build_feature_sets
from src.model_hmm_gmm import _build_model, _regularize_covars
from src.splits import split_by_years
from src.utils import set_seed


def _fold_length_years(fold_id: str) -> int:
    parts = fold_id.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected fold_id: {fold_id}")
    start = int(parts[0])
    end = int(parts[1])
    return end - start + 1


def _component_assignments(model, X: np.ndarray, states: np.ndarray) -> np.ndarray:
    n_mix = int(getattr(model, "n_mix", 1))
    if n_mix <= 1:
        return np.zeros(states.shape[0], dtype=int)

    comp = np.full(states.shape[0], -1, dtype=int)
    posts = state_component_posteriors(model, X, states)
    for _, (idx, resp) in posts.items():
        if resp.size == 0:
            continue
        comp[idx] = np.argmax(resp, axis=1)
    return comp


def _load_summary(path: Path) -> List[Dict[str, object]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected list in summary: {path}")
    return records


def _run_for_record(
    record: Dict[str, object],
    cfg: dict,
    df: pd.DataFrame,
    feature_cols: List[str],
    out_root: Path,
    train_only: bool,
    delete_old: bool,
    fit_full: bool,
) -> Path:
    fold_id = str(record["fold_id"])
    k = int(record["k"])
    n_mix = int(record["n_mix"])
    cov_type = str(record["cov_type"])
    seed = int(record["seed"])

    # Split data
    train_years = list(range(int(fold_id.split("_")[0]), int(fold_id.split("_")[1]) + 1))
    test_year = int(fold_id.split("_")[3])
    train_df, test_df = split_by_years(df, cfg["data"]["timestamp_col"], train_years, [test_year])
    if fit_full:
        full_years = train_years + [test_year]
        train_df = df[df[cfg["data"]["timestamp_col"]].dt.year.isin(full_years)].copy()

    # Preprocess for model fitting
    if fit_full:
        _, _, X_train, X_test = _preprocess(train_df, train_df, feature_cols, cfg)
    else:
        _, _, X_train, X_test = _preprocess(train_df, test_df, feature_cols, cfg)

    # Fit model and predict states/components
    hmm_cfg = cfg["models"]["hmm"]
    model = _build_model(k, cov_type, n_mix, seed, hmm_cfg)
    model.fit(X_train)

    reg_list = hmm_cfg.get("reg_covar_list") or [
        hmm_cfg.get("reg_covar", hmm_cfg.get("min_covar", 1.0e-6))
    ]
    last_exc: Exception | None = None
    for reg_covar in reg_list:
        _regularize_covars(model, float(reg_covar))
        try:
            # Validate before decoding
            _ = model.score(X_train)
            if not fit_full:
                _ = model.score(X_test)
            viterbi_train = model.predict(X_train)
            viterbi_test = model.predict(X_test)
            break
        except Exception as exc:
            last_exc = exc
            viterbi_train = None
            viterbi_test = None
            continue
    if viterbi_train is None or viterbi_test is None:
        raise RuntimeError(f"HMM decode failed after regularization: {last_exc}")

    comp_train = _component_assignments(model, X_train, viterbi_train)
    comp_test = _component_assignments(model, X_test, viterbi_test)

    train_start = pd.Timestamp(train_df[cfg["data"]["timestamp_col"]].min())
    train_end = pd.Timestamp(train_df[cfg["data"]["timestamp_col"]].max())
    test_start = pd.Timestamp(test_df[cfg["data"]["timestamp_col"]].min())
    test_end = pd.Timestamp(test_df[cfg["data"]["timestamp_col"]].max())

    # Build output
    train_out = train_df.copy()
    train_out["state"] = viterbi_train
    train_out["component"] = comp_train
    train_out["fold_id"] = fold_id
    if not train_only:
        train_out["split"] = "train"
    train_out["train_start"] = train_start
    train_out["train_end"] = train_end
    train_out["test_start"] = test_start
    train_out["test_end"] = test_end

    test_out = None
    if not train_only and not fit_full:
        test_out = test_df.copy()
        test_out["state"] = viterbi_test
        test_out["component"] = comp_test
        test_out["fold_id"] = fold_id
        test_out["split"] = "test"
        test_out["train_start"] = train_start
        test_out["train_end"] = train_end
        test_out["test_start"] = test_start
        test_out["test_end"] = test_end

    if train_only:
        out_df = train_out
    else:
        out_df = pd.concat([train_out, test_out], ignore_index=True)

    fold_years = _fold_length_years(fold_id)
    dest_dir = out_root / f"{fold_years}y" / f"k{k}m{n_mix}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    if train_only:
        out_name = f"candles_12h_trainonly_{fold_id}.csv"
    else:
        out_name = f"candles_12h_{fold_id}.csv"
    out_path = dest_dir / out_name
    if delete_old:
        old_path = dest_dir / f"candles_12h_{fold_id}.csv"
        if old_path.exists() and old_path != out_path:
            try:
                old_path.unlink()
            except Exception:
                pass
    out_df.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export candles with OHLCV+indicators + state/component per fold/config."
    )
    parser.add_argument(
        "--config",
        default="pipeline/step_regime_hmm_gmm/config_12h_k2k3_m1m3.yaml",
        help="Path to config yaml.",
    )
    parser.add_argument(
        "--summary",
        default="pipeline/step_regime_hmm_gmm/results/12h/raw_v2/first_pack_summary.json",
        help="Path to first_pack_summary.json (source of fold/config/seed).",
    )
    parser.add_argument(
        "--out-root",
        default="pipeline/step_regime_hmm_gmm/results/12h",
        help="Output root for 3y/4y/5y folders.",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Write train rows only (no test rows).",
    )
    parser.add_argument(
        "--delete-old",
        action="store_true",
        help="Delete old candles_12h_{fold}.csv when writing train-only files.",
    )
    parser.add_argument(
        "--fit-full",
        action="store_true",
        help="Fit model on train+test years and output a single train-only file.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    df = load_data(
        cfg["data"]["path"],
        cfg["data"]["timestamp_col"],
        drop_cols=cfg["data"].get("drop_cols"),
        start_date=cfg["data"].get("start_date"),
        end_date=cfg["data"].get("end_date"),
    )

    feature_sets = build_feature_sets(df, cfg)
    feature_cols = feature_sets.get("indicators_only")
    if not feature_cols:
        raise RuntimeError("indicators_only feature set not found.")

    summary_path = Path(args.summary)
    records = _load_summary(summary_path)

    out_root = Path(args.out_root)
    written: List[Path] = []
    for record in records:
        if str(record.get("feature_set", "")) != "indicators_only":
            continue
        written.append(
            _run_for_record(
                record,
                cfg,
                df,
                feature_cols,
                out_root,
                args.train_only,
                args.delete_old,
                args.fit_full,
            )
        )

    print(f"written={len(written)}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_ledgers(
    ledger_dir: Path,
    gmm_sweep_all: pd.DataFrame | None = None,
    gmm_top_per_fold: pd.DataFrame | None = None,
    ensemble_weights: pd.DataFrame | None = None,
    ensemble_summary: pd.DataFrame | None = None,
) -> None:
    ledger_dir.mkdir(parents=True, exist_ok=True)
    if gmm_sweep_all is not None:
        gmm_sweep_all.to_csv(ledger_dir / "gmm_sweep_all.csv", index=False)
    if gmm_top_per_fold is not None:
        gmm_top_per_fold.to_csv(ledger_dir / "gmm_top_per_fold.csv", index=False)
    if ensemble_weights is not None:
        ensemble_weights.to_csv(ledger_dir / "ensemble_weights.csv", index=False)
    if ensemble_summary is not None:
        ensemble_summary.to_csv(ledger_dir / "ensemble_summary.csv", index=False)


def save_fold_candles(df: pd.DataFrame, out_path: Path, float_format: str = "%.8f") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, float_format=float_format)

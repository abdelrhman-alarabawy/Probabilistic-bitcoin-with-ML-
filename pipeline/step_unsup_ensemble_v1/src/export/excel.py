from __future__ import annotations

from pathlib import Path

import pandas as pd


def _safe_read(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def build_excel_summary(root_dir: Path) -> Path:
    ledger_dir = root_dir / "ledger"
    reports_dir = root_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary_xlsx = reports_dir / "ensemble_summary.xlsx"

    sweep_df = _safe_read(ledger_dir / "gmm_sweep_all.csv")
    top_df = _safe_read(ledger_dir / "gmm_top_per_fold.csv")
    weights_df = _safe_read(ledger_dir / "ensemble_weights.csv")
    summary_df = _safe_read(ledger_dir / "ensemble_summary.csv")
    dist_df = _safe_read(ledger_dir / "ensemble_distribution.csv")

    with pd.ExcelWriter(summary_xlsx, engine="openpyxl") as writer:
        sweep_df.to_excel(writer, sheet_name="gmm_sweep_all", index=False)
        top_df.to_excel(writer, sheet_name="gmm_top_per_fold", index=False)
        weights_df.to_excel(writer, sheet_name="ensemble_weights", index=False)
        summary_df.to_excel(writer, sheet_name="ensemble_summary", index=False)
        dist_df.to_excel(writer, sheet_name="distributions", index=False)

    return summary_xlsx

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import load_config
from .utils import parse_model_id, setup_logging


logger = setup_logging()


def build_excel_summary(config_path: str | Path) -> dict:
    cfg = load_config(config_path)

    output_root = Path(cfg["output"]["root_dir"])
    labeled_dir = output_root / "labeled"
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not labeled_dir.exists():
        raise FileNotFoundError(f"Labeled directory not found: {labeled_dir}. Run label-top first.")

    rows = []
    for path in labeled_dir.glob("*__labeled.csv"):
        df = pd.read_csv(path)
        if cfg["data"]["label_col"] not in df.columns:
            raise ValueError(f"Missing label column in {path}")

        counts = df[cfg["data"]["label_col"]].value_counts(dropna=False)
        n_total = len(df)
        n_long = int(counts.get("long", 0))
        n_short = int(counts.get("short", 0))
        n_skip = int(counts.get("skip", 0))

        model_id = path.stem.replace("__labeled", "")
        meta = parse_model_id(model_id)

        rows.append(
            {
                "model_id": model_id,
                "fold_id": meta.get("fold_id"),
                "featureset": meta.get("featureset"),
                "K": meta.get("K"),
                "covariance_type": meta.get("covariance_type"),
                "reg_covar": meta.get("reg_covar"),
                "n_total": n_total,
                "n_long": n_long,
                "n_short": n_short,
                "n_skip": n_skip,
                "long_share": n_long / n_total if n_total else 0.0,
                "short_share": n_short / n_total if n_total else 0.0,
                "skip_share": n_skip / n_total if n_total else 0.0,
            }
        )

    summary_df = pd.DataFrame(rows).sort_values(["fold_id", "model_id"])

    summary_csv = reports_dir / "label_counts_top10.csv"
    summary_xlsx = reports_dir / "label_counts_top10.xlsx"

    summary_df.to_csv(summary_csv, index=False)

    with pd.ExcelWriter(summary_xlsx, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        if not summary_df.empty:
            by_fold = summary_df.pivot_table(
                index="fold_id",
                values=["n_long", "n_short", "n_skip", "n_total"],
                aggfunc="sum",
            )
            by_fold.to_excel(writer, sheet_name="by_fold")

    logger.info("Wrote summary to %s", summary_xlsx)
    return {"summary_csv": summary_csv, "summary_xlsx": summary_xlsx}

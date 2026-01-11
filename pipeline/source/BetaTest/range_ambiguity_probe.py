from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


LABELS_ALLOWED = {"long", "short", "skip"}


def _find_column(df: pd.DataFrame, target: str) -> Optional[str]:
    lower_map = {col.lower(): col for col in df.columns}
    return lower_map.get(target.lower())


def _find_ambig_column(df: pd.DataFrame, ambig_col: str) -> Tuple[Optional[str], List[str]]:
    if ambig_col in df.columns:
        return ambig_col, []
    candidates = [col for col in df.columns if "ambig" in col.lower()]
    return None, candidates


def run_range_ambiguity_probe(
    df: pd.DataFrame,
    label_col: str,
    ambig_col: str,
    tables_dir: Path,
    min_range_pcts: List[float],
) -> Dict[str, object]:
    if label_col not in df.columns:
        raise ValueError(f"Missing label column '{label_col}' for cleaning probe.")

    open_col = _find_column(df, "open")
    high_col = _find_column(df, "high")
    low_col = _find_column(df, "low")
    close_col = _find_column(df, "close")
    volume_col = _find_column(df, "volume")
    available = [col for col in [open_col, high_col, low_col, close_col, volume_col] if col]

    print(f"OHLCV columns detected: {available}")

    ambig_name, ambig_candidates = _find_ambig_column(df, ambig_col)
    if ambig_name:
        print(f"ambiguous_flag column found: {ambig_name}")
    else:
        print("ambiguous_flag column missing.")
        if ambig_candidates:
            print(f"Ambiguous candidates: {ambig_candidates}")

    label_values = set(df[label_col].dropna().unique())
    unexpected = label_values - LABELS_ALLOWED
    if unexpected:
        print(f"WARNING: Unexpected labels found: {unexpected}")

    for col in (open_col, high_col, low_col):
        if col is None:
            raise ValueError("Missing required OHLC columns for range_pct probe.")

    open_series = pd.to_numeric(df[open_col], errors="coerce").replace(0, pd.NA)
    high_series = pd.to_numeric(df[high_col], errors="coerce")
    low_series = pd.to_numeric(df[low_col], errors="coerce")
    range_pct = (high_series - low_series) / open_series

    quantiles = {
        "min": range_pct.min(skipna=True),
        "p1": range_pct.quantile(0.01),
        "p5": range_pct.quantile(0.05),
        "p10": range_pct.quantile(0.10),
        "p25": range_pct.quantile(0.25),
        "p50": range_pct.quantile(0.50),
        "p75": range_pct.quantile(0.75),
        "p90": range_pct.quantile(0.90),
        "p95": range_pct.quantile(0.95),
        "p99": range_pct.quantile(0.99),
        "max": range_pct.max(skipna=True),
    }

    quant_df = pd.DataFrame(
        [{"quantile": key, "value": float(val) if pd.notna(val) else None} for key, val in quantiles.items()]
    )
    quant_path = tables_dir / "range_pct_quantiles.csv"
    quant_df.to_csv(quant_path, index=False)

    print("Range_pct quantiles:")
    for key in ["min", "p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "max"]:
        print(f"  {key}: {quantiles[key]}")

    simulation_rows = []
    for min_range_pct in min_range_pcts:
        df_copy = df.copy()
        original_labels = df_copy[label_col].copy()

        changed_ambig = 0
        if ambig_name:
            ambig_mask = df_copy[ambig_name].astype(bool)
            changed_ambig = int((ambig_mask & (df_copy[label_col] != "skip")).sum())
            df_copy.loc[ambig_mask, label_col] = "skip"

        range_mask = (range_pct < min_range_pct).fillna(False)
        changed_minrange = int((range_mask & (df_copy[label_col] != "skip")).sum())
        df_copy.loc[range_mask, label_col] = "skip"

        total_changed = int((df_copy[label_col] != original_labels).sum())
        counts = df_copy[label_col].value_counts().to_dict()
        simulation_rows.append(
            {
                "min_range_pct": min_range_pct,
                "changed_ambig": changed_ambig,
                "changed_minrange": changed_minrange,
                "total_changed": total_changed,
                "counts_long": int(counts.get("long", 0)),
                "counts_short": int(counts.get("short", 0)),
                "counts_skip": int(counts.get("skip", 0)),
            }
        )

        print(
            f"min_range_pct={min_range_pct} changed_ambig={changed_ambig} "
            f"changed_minrange={changed_minrange} total_changed={total_changed}"
        )

    sim_df = pd.DataFrame(simulation_rows)
    sim_path = tables_dir / "cleaning_impact_simulation.csv"
    sim_df.to_csv(sim_path, index=False)

    print("==== CLEANING PROBE ====")
    print(f"ambiguous_col: {ambig_name if ambig_name else 'NONE'}")
    print(
        f"range_pct p50={quantiles['p50']} p90={quantiles['p90']} p95={quantiles['p95']}"
    )
    print(f"{sim_path}")
    print("========================")

    return {
        "ambig_col": ambig_name,
        "quantiles": quantiles,
        "quantiles_path": quant_path,
        "simulation_path": sim_path,
    }

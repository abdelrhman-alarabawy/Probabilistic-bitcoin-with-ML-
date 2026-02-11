from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
from typing import Dict, Optional, Tuple

import pandas as pd


def _prepare_label_input(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    work = df.copy()
    if timestamp_col != "timestamp":
        work = work.rename(columns={timestamp_col: "timestamp"})

    keep_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [col for col in keep_cols if col not in work.columns]
    if missing:
        raise ValueError(f"Label script requires OHLCV columns; missing: {missing}")

    out = work.loc[:, keep_cols].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return out


def _read_script_output(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected label output not found: {path}")
    df = pd.read_csv(path)
    if "ts_utc" in df.columns and "timestamp" not in df.columns:
        df = df.rename(columns={"ts_utc": "timestamp"})
    if "timestamp" not in df.columns:
        raise ValueError(f"Label output missing timestamp column: {path}")

    label_col = None
    for candidate in ["candle_type", "label", "signal", "side", "trade_label"]:
        if candidate in df.columns:
            label_col = candidate
            break
    if label_col is None:
        raise ValueError(f"No label column found in label output: {path}")

    out = df.loc[:, ["timestamp", label_col]].copy()
    out = out.rename(columns={label_col: "trade_label"})
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    out["trade_label"] = out["trade_label"].astype(str).str.strip().str.lower()
    return out


def run_label_script_subprocess(
    script_path: Path,
    input_ohlcv_df: pd.DataFrame,
    output_mode: str = "no5m",
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if not script_path.exists():
        raise FileNotFoundError(f"Label script not found: {script_path}")

    with tempfile.TemporaryDirectory(prefix="gmm_label_") as tmp:
        tmp_dir = Path(tmp)
        temp_input = tmp_dir / "data_12h_indicators.csv"
        input_ohlcv_df.to_csv(temp_input, index=False)

        cmd = ["python", str(script_path)]
        proc = subprocess.run(
            cmd,
            cwd=tmp_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        diagnostics: Dict[str, object] = {
            "label_script_command": cmd,
            "label_script_returncode": proc.returncode,
            "label_script_stdout_tail": proc.stdout[-3000:],
            "label_script_stderr_tail": proc.stderr[-3000:],
            "label_mode_selected": output_mode,
        }
        if proc.returncode != 0:
            raise RuntimeError(
                "Label script failed with return code "
                f"{proc.returncode}. stderr tail: {proc.stderr[-1000:]}"
            )

        baseline_path = tmp_dir / "pipeline" / "source" / "labeling" / "output_12h_labels_baseline_no5m.csv"
        with5m_path = tmp_dir / "pipeline" / "source" / "labeling" / "output_12h_labels_with5m.csv"

        chosen_path = baseline_path if output_mode == "no5m" else with5m_path
        labels_df = _read_script_output(chosen_path)

    return labels_df, diagnostics


def has_required_ohlcv(df: pd.DataFrame) -> bool:
    required = {"open", "high", "low", "close", "volume"}
    cols = {c.lower() for c in df.columns}
    return required.issubset(cols)


def build_group_labels(
    df: pd.DataFrame,
    timestamp_col: Optional[str],
    script_path: Path,
    label_mode: str,
) -> Tuple[Optional[pd.DataFrame], Dict[str, object]]:
    if timestamp_col is None:
        return None, {"labeling_enabled": False, "label_warning": "Missing timestamp column; labeling skipped."}

    if not has_required_ohlcv(df):
        return None, {"labeling_enabled": False, "label_warning": "Missing OHLCV columns; labeling skipped."}

    ohlcv_df = _prepare_label_input(df, timestamp_col=timestamp_col)
    labels_df, script_diag = run_label_script_subprocess(
        script_path=script_path,
        input_ohlcv_df=ohlcv_df,
        output_mode=label_mode,
    )
    diag = {"labeling_enabled": True, "label_rows": int(labels_df.shape[0]), **script_diag}
    return labels_df, diag

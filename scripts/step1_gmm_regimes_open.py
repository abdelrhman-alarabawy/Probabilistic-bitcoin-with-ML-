from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler


# === CONFIGURATION ===
INPUT_CSV_CANDIDATES = [
    Path(r"D:\GitHub\bitcoin-probabilistic-learning\data\processed\12h_features_indicators_with_ohlcv.csv"),
    Path("data/processed/12h_features_indicators_with_ohlcv.csv"),
]
OUTPUT_DIR = Path(
    r"D:\GitHub\bitcoin-probabilistic-learning\pipeline\step_regime_gmm_12h_open\results"
)

MODE = "fit_full"  # "fit_full" or "fit_initial_then_apply"
INITIAL_TRAIN_MONTHS = 18
SHIFT_PERIODS = 1

K_RANGE = list(range(2, 9))
COVARIANCE_TYPES = ["full", "diag"]
NA_THRESHOLD = 0.30
RANDOM_STATE = 42

PROB_THRESHOLDS = [0.6, 0.7, 0.8, 0.9]


META_COLS = {"timestamp", "open", "high", "low", "close", "volume"}
META_COL_ORDER = ["timestamp", "open", "high", "low", "close", "volume"]
LABEL_HINTS = {"candle_type", "label", "label_ambiguous", "target"}


def resolve_input_path(candidates: Iterable[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Input CSV not found. Tried: {[str(p) for p in candidates]}")


def find_timestamp_column(columns: Iterable[str]) -> str:
    if "timestamp" in columns:
        return "timestamp"
    for candidate in ("ts_utc", "time", "open_time", "datetime"):
        if candidate in columns:
            return candidate
    raise ValueError("Timestamp column not found. Expected 'timestamp' or common alternatives.")


def parse_timestamp_series(series: pd.Series) -> pd.Series:
    if np.issubdtype(series.dtype, np.number):
        return parse_numeric_timestamp(series)
    parsed = pd.to_datetime(series, utc=True, errors="coerce")
    if parsed.isna().mean() > 0.05:
        as_num = pd.to_numeric(series, errors="coerce")
        if as_num.notna().any():
            parsed = parse_numeric_timestamp(as_num)
    return parsed


def parse_numeric_timestamp(series: pd.Series) -> pd.Series:
    max_val = series.max()
    if max_val > 1e12:
        return pd.to_datetime(series, unit="ms", utc=True, errors="coerce")
    if max_val > 1e10:
        return pd.to_datetime(series, unit="s", utc=True, errors="coerce")
    return pd.to_datetime(series, utc=True, errors="coerce")


def compute_entropy(prob_matrix: np.ndarray) -> np.ndarray:
    return -np.sum(prob_matrix * np.log(prob_matrix + 1e-12), axis=1)


def compute_run_lengths(regimes: np.ndarray) -> Tuple[List[int], Dict[int, List[int]]]:
    run_lengths: List[int] = []
    per_regime: Dict[int, List[int]] = {}
    if len(regimes) == 0:
        return run_lengths, per_regime
    current = regimes[0]
    length = 1
    for regime in regimes[1:]:
        if regime == current:
            length += 1
        else:
            run_lengths.append(length)
            per_regime.setdefault(current, []).append(length)
            current = regime
            length = 1
    run_lengths.append(length)
    per_regime.setdefault(current, []).append(length)
    return run_lengths, per_regime


def df_to_markdown(df: pd.DataFrame, floatfmt: str = ".6f", index: bool = True) -> str:
    def format_value(value: object) -> str:
        if pd.isna(value):
            return "nan"
        if isinstance(value, (float, np.floating)):
            return format(value, floatfmt)
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        return str(value)

    if index:
        index_name = df.index.name if df.index.name else "index"
        columns = [index_name] + [str(col) for col in df.columns]
        rows = []
        for idx, row in df.iterrows():
            rows.append([format_value(idx)] + [format_value(v) for v in row.tolist()])
    else:
        columns = [str(col) for col in df.columns]
        rows = [[format_value(v) for v in row] for row in df.values.tolist()]

    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    input_path = resolve_input_path(INPUT_CSV_CANDIDATES)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figures_dir = OUTPUT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading input: {input_path}")
    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()

    ts_col = find_timestamp_column(df.columns)
    if ts_col != "timestamp":
        df = df.rename(columns={ts_col: "timestamp"})

    df["timestamp"] = parse_timestamp_series(df["timestamp"])
    bad_ratio = df["timestamp"].isna().mean()
    if bad_ratio > 0.01:
        raise ValueError(f"Timestamp parsing failed. NaT ratio={bad_ratio:.2%}")
    if df["timestamp"].isna().any():
        df = df[df["timestamp"].notna()].copy()

    missing_meta = META_COLS - set(df.columns)
    if missing_meta:
        raise ValueError(f"Missing required OHLCV columns: {sorted(missing_meta)}")

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("Timestamps are not strictly increasing after sorting.")

    label_cols = [
        col
        for col in df.columns
        if col.lower() in LABEL_HINTS
        or "label" in col.lower()
        or "candle_type" in col.lower()
        or "regime" in col.lower()
    ]

    candidate_cols = [col for col in df.columns if col not in META_COLS and col not in label_cols]
    for col in candidate_cols:
        if not np.issubdtype(df[col].dtype, np.number):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    numeric_feature_cols = [
        col for col in candidate_cols if np.issubdtype(df[col].dtype, np.number)
    ]

    if not numeric_feature_cols:
        raise ValueError("No numeric feature columns found after filtering.")

    df[numeric_feature_cols] = df[numeric_feature_cols].shift(SHIFT_PERIODS)
    df = df.iloc[SHIFT_PERIODS:].copy()

    na_ratio = df[numeric_feature_cols].isna().mean()
    numeric_feature_cols = [
        col for col in numeric_feature_cols if na_ratio[col] <= NA_THRESHOLD
    ]
    if not numeric_feature_cols:
        raise ValueError("All feature columns dropped due to NaN threshold.")

    nunique = df[numeric_feature_cols].nunique(dropna=True)
    numeric_feature_cols = [col for col in numeric_feature_cols if nunique[col] > 1]
    if not numeric_feature_cols:
        raise ValueError("All feature columns dropped as constant.")

    X_raw = df[numeric_feature_cols]

    if MODE not in {"fit_full", "fit_initial_then_apply"}:
        raise ValueError(f"Unknown MODE '{MODE}'.")

    if MODE == "fit_initial_then_apply":
        start_ts = df["timestamp"].min()
        train_end = start_ts + pd.DateOffset(months=INITIAL_TRAIN_MONTHS)
        train_mask = df["timestamp"] < train_end
        if train_mask.sum() < max(100, len(df) * 0.1):
            raise ValueError("Initial training window too small for stable GMM fitting.")
    else:
        train_mask = pd.Series(True, index=df.index)

    imputer = SimpleImputer(strategy="median")
    train_na_ratio = X_raw[train_mask].isna().mean()
    all_nan_cols = train_na_ratio[train_na_ratio == 1.0].index.tolist()
    if all_nan_cols:
        numeric_feature_cols = [col for col in numeric_feature_cols if col not in all_nan_cols]
        if not numeric_feature_cols:
            raise ValueError("All feature columns are NaN in the training window.")
        X_raw = df[numeric_feature_cols]

    train_nunique = X_raw[train_mask].nunique(dropna=True)
    constant_train_cols = train_nunique[train_nunique <= 1].index.tolist()
    if constant_train_cols:
        numeric_feature_cols = [
            col for col in numeric_feature_cols if col not in constant_train_cols
        ]
        if not numeric_feature_cols:
            raise ValueError("All feature columns are constant in the training window.")
        X_raw = df[numeric_feature_cols]

    X_train_imputed = imputer.fit_transform(X_raw[train_mask])
    X_full_imputed = imputer.transform(X_raw)

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_full_scaled = scaler.transform(X_full_imputed)

    selection_results = []
    print("Model selection: scanning K and covariance_type")
    for cov_type in COVARIANCE_TYPES:
        for k in K_RANGE:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type=cov_type,
                random_state=RANDOM_STATE,
                n_init=5,
                max_iter=500,
                reg_covar=1e-6,
            )
            gmm.fit(X_train_scaled)
            bic = gmm.bic(X_train_scaled)
            aic = gmm.aic(X_train_scaled)
            probs = gmm.predict_proba(X_train_scaled)
            max_prob = probs.max(axis=1)
            entropy = compute_entropy(probs)
            selection_results.append(
                {
                    "k": k,
                    "covariance_type": cov_type,
                    "bic": bic,
                    "aic": aic,
                    "avg_max_prob": float(np.mean(max_prob)),
                    "avg_entropy": float(np.mean(entropy)),
                }
            )
            print(
                f"  k={k} cov={cov_type} BIC={bic:.2f} AIC={aic:.2f} "
                f"avg_pmax={np.mean(max_prob):.3f} avg_entropy={np.mean(entropy):.3f}"
            )

    selection_df = pd.DataFrame(selection_results).sort_values("bic").reset_index(drop=True)
    best_row = selection_df.iloc[0]
    best_k = int(best_row["k"])
    best_cov = str(best_row["covariance_type"])
    print(f"Selected model: k={best_k} cov={best_cov} (lowest BIC)")

    best_gmm = GaussianMixture(
        n_components=best_k,
        covariance_type=best_cov,
        random_state=RANDOM_STATE,
        n_init=5,
        max_iter=500,
        reg_covar=1e-6,
    )
    best_gmm.fit(X_train_scaled)

    probs_full = best_gmm.predict_proba(X_full_scaled)
    regime_id = probs_full.argmax(axis=1)
    regime_prob_max = probs_full.max(axis=1)
    entropy_full = compute_entropy(probs_full)

    df_out = df[META_COL_ORDER].copy()
    df_out[numeric_feature_cols] = X_full_imputed
    df_out["regime_id"] = regime_id
    df_out["regime_prob_max"] = regime_prob_max
    df_out["entropy"] = entropy_full

    output_csv = OUTPUT_DIR / "gmm_regimes_per_row.csv"
    df_out.to_csv(output_csv, index=False)

    joblib.dump(best_gmm, OUTPUT_DIR / "gmm_model.joblib")
    joblib.dump(scaler, OUTPUT_DIR / "scaler.joblib")
    joblib.dump(imputer, OUTPUT_DIR / "imputer.joblib")
    with (OUTPUT_DIR / "selected_features.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "selected_features": numeric_feature_cols,
                "meta_cols": sorted(META_COLS),
                "label_cols": label_cols,
                "shift_periods": SHIFT_PERIODS,
                "na_threshold": NA_THRESHOLD,
                "mode": MODE,
                "initial_train_months": INITIAL_TRAIN_MONTHS,
            },
            f,
            indent=2,
        )

    # === VALIDATION ===
    prob_thresholds = {
        thr: float(np.mean(regime_prob_max >= thr)) for thr in PROB_THRESHOLDS
    }

    n_regimes = best_k
    transition = pd.crosstab(
        regime_id[:-1],
        regime_id[1:],
        rownames=["from"],
        colnames=["to"],
        dropna=False,
    ).reindex(index=range(n_regimes), columns=range(n_regimes), fill_value=0)

    run_lengths, per_regime_runs = compute_run_lengths(regime_id)
    avg_run_by_regime = {
        regime: float(np.mean(lengths)) if lengths else 0.0
        for regime, lengths in per_regime_runs.items()
    }

    r_next = (df["close"] - df["open"]) / df["open"]
    df_returns = pd.DataFrame({"regime_id": regime_id, "r_next": r_next})

    returns_summary = []
    for regime in range(n_regimes):
        subset = df_returns[df_returns["regime_id"] == regime]["r_next"]
        if subset.empty:
            returns_summary.append(
                {
                    "regime": regime,
                    "count": 0,
                    "mean": np.nan,
                    "median": np.nan,
                    "std": np.nan,
                    "sharpe_like": np.nan,
                    "win_rate": np.nan,
                    "q05": np.nan,
                    "q25": np.nan,
                    "q50": np.nan,
                    "q75": np.nan,
                    "q95": np.nan,
                }
            )
            continue
        mean_val = subset.mean()
        std_val = subset.std(ddof=0)
        sharpe_like = mean_val / std_val if std_val > 0 else np.nan
        returns_summary.append(
            {
                "regime": regime,
                "count": int(subset.shape[0]),
                "mean": mean_val,
                "median": subset.median(),
                "std": std_val,
                "sharpe_like": sharpe_like,
                "win_rate": float((subset > 0).mean()),
                "q05": subset.quantile(0.05),
                "q25": subset.quantile(0.25),
                "q50": subset.quantile(0.50),
                "q75": subset.quantile(0.75),
                "q95": subset.quantile(0.95),
            }
        )

    returns_summary_df = pd.DataFrame(returns_summary)

    feature_df = pd.DataFrame(X_full_imputed, columns=numeric_feature_cols, index=df.index)
    global_mean = feature_df.mean()
    global_std = feature_df.std(ddof=0).replace(0, np.nan)

    feature_profiles: Dict[int, pd.DataFrame] = {}
    for regime in range(n_regimes):
        subset = feature_df[regime_id == regime]
        if subset.empty:
            feature_profiles[regime] = pd.DataFrame(
                columns=["feature", "regime_mean", "global_mean", "z_diff"]
            )
            continue
        regime_mean = subset.mean()
        z_diff = (regime_mean - global_mean) / global_std
        top = z_diff.abs().sort_values(ascending=False).head(15).index
        profile = pd.DataFrame(
            {
                "feature": top,
                "regime_mean": regime_mean[top].values,
                "global_mean": global_mean[top].values,
                "z_diff": z_diff[top].values,
            }
        )
        feature_profiles[regime] = profile

    candle_type_col = None
    for col in df.columns:
        if col.lower() == "candle_type":
            candle_type_col = col
            break

    candle_distribution = None
    candle_distribution_norm = None
    if candle_type_col is not None:
        candle_distribution = pd.crosstab(regime_id, df[candle_type_col])
        candle_distribution_norm = pd.crosstab(
            regime_id, df[candle_type_col], normalize="index"
        )

    # === FIGURES ===
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for cov_type in COVARIANCE_TYPES:
        subset = selection_df[selection_df["covariance_type"] == cov_type]
        axes[0].plot(subset["k"], subset["bic"], marker="o", label=cov_type)
        axes[1].plot(subset["k"], subset["aic"], marker="o", label=cov_type)
    axes[0].set_title("BIC vs K")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("BIC")
    axes[1].set_title("AIC vs K")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("AIC")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "bic_aic_vs_k.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(regime_prob_max, bins=30, color="#2c7fb8", alpha=0.8)
    ax.set_title("Max posterior probability")
    ax.set_xlabel("regime_prob_max")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "probmax_hist.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(entropy_full, bins=30, color="#7fcdbb", alpha=0.8)
    ax.set_title("Entropy of posterior")
    ax.set_xlabel("entropy")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "entropy_hist.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 2))
    ax.imshow(regime_id[np.newaxis, :], aspect="auto", cmap="tab20")
    ax.set_yticks([])
    ax.set_title("Regime timeline")
    ticks = np.linspace(0, len(df) - 1, 6, dtype=int)
    ax.set_xticks(ticks)
    tick_labels = [
        df["timestamp"].iloc[i].strftime("%Y-%m") if 0 <= i < len(df) else ""
        for i in ticks
    ]
    ax.set_xticklabels(tick_labels, rotation=0)
    fig.tight_layout()
    fig.savefig(figures_dir / "regime_timeline.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(transition.values, cmap="Blues")
    ax.set_title("Regime transition counts")
    ax.set_xlabel("to")
    ax.set_ylabel("from")
    ax.set_xticks(range(n_regimes))
    ax.set_yticks(range(n_regimes))
    for i in range(n_regimes):
        for j in range(n_regimes):
            ax.text(j, i, str(int(transition.iloc[i, j])), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(figures_dir / "regime_transition_heatmap.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    data = [df_returns[df_returns["regime_id"] == r]["r_next"] for r in range(n_regimes)]
    ax.boxplot(data, labels=[str(r) for r in range(n_regimes)])
    ax.set_title("Next-candle return by regime")
    ax.set_xlabel("regime")
    ax.set_ylabel("r_next")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "regime_return_boxplot.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(run_lengths, bins=30, color="#feb24c", alpha=0.8)
    ax.set_title("Run length distribution")
    ax.set_xlabel("run_length")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "run_length_hist.png", dpi=150)
    plt.close(fig)

    # === REPORT ===
    report_lines: List[str] = []
    report_lines.append("# GMM regime discovery validation report")
    report_lines.append("")
    report_lines.append("## Configuration")
    report_lines.append(f"- input_csv: {input_path}")
    report_lines.append(f"- mode: {MODE}")
    report_lines.append(f"- shift_periods: {SHIFT_PERIODS} (features shifted by 1 for OPEN entry)")
    report_lines.append(f"- initial_train_months: {INITIAL_TRAIN_MONTHS}")
    report_lines.append(f"- rows_used: {len(df)}")
    report_lines.append(f"- features_used: {len(numeric_feature_cols)}")
    report_lines.append("")

    report_lines.append("## Model selection (top 3 by BIC)")
    report_lines.append("| rank | k | covariance | BIC | AIC | avg_max_prob | avg_entropy |")
    report_lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for idx, row in selection_df.head(3).iterrows():
        report_lines.append(
            f"| {idx + 1} | {int(row['k'])} | {row['covariance_type']} | "
            f"{row['bic']:.2f} | {row['aic']:.2f} | {row['avg_max_prob']:.4f} | "
            f"{row['avg_entropy']:.4f} |"
        )
    report_lines.append("")
    report_lines.append(f"Selected model: k={best_k}, covariance={best_cov}")
    report_lines.append("")

    report_lines.append("## Separation diagnostics")
    report_lines.append(f"- avg max posterior prob: {np.mean(regime_prob_max):.4f}")
    report_lines.append(f"- avg entropy: {np.mean(entropy_full):.4f}")
    for thr, pct in prob_thresholds.items():
        report_lines.append(f"- pct prob_max >= {thr:.1f}: {pct:.2%}")
    report_lines.append("")

    report_lines.append("## Regime stability")
    report_lines.append("Transition matrix (counts):")
    report_lines.append("")
    report_lines.append(df_to_markdown(transition, floatfmt=".0f", index=True))
    report_lines.append("")
    report_lines.append("Average run length by regime:")
    report_lines.append("")
    report_lines.append("| regime | avg_run_length |")
    report_lines.append("| --- | --- |")
    for regime in range(n_regimes):
        avg_run = avg_run_by_regime.get(regime, 0.0)
        report_lines.append(f"| {regime} | {avg_run:.2f} |")
    report_lines.append("")

    report_lines.append("## Economic interpretability (next-candle returns)")
    report_lines.append(df_to_markdown(returns_summary_df, floatfmt=".6f", index=False))
    report_lines.append("")

    report_lines.append("## Feature profile (top 15 by abs z-diff)")
    for regime in range(n_regimes):
        report_lines.append(f"### Regime {regime}")
        profile = feature_profiles[regime]
        if profile.empty:
            report_lines.append("No samples in this regime.")
            report_lines.append("")
            continue
        report_lines.append(df_to_markdown(profile, floatfmt=".6f", index=False))
        report_lines.append("")

    if candle_type_col is not None:
        report_lines.append("## Candle type distribution (optional)")
        report_lines.append("Counts by regime:")
        report_lines.append("")
        report_lines.append(df_to_markdown(candle_distribution, floatfmt=".0f", index=True))
        report_lines.append("")
        report_lines.append("Row-normalized distribution by regime:")
        report_lines.append("")
        report_lines.append(df_to_markdown(candle_distribution_norm, floatfmt=".4f", index=True))
        report_lines.append("")

    report_lines.append("## Figures")
    report_lines.append(f"- {figures_dir / 'bic_aic_vs_k.png'}")
    report_lines.append(f"- {figures_dir / 'probmax_hist.png'}")
    report_lines.append(f"- {figures_dir / 'entropy_hist.png'}")
    report_lines.append(f"- {figures_dir / 'regime_timeline.png'}")
    report_lines.append(f"- {figures_dir / 'regime_transition_heatmap.png'}")
    report_lines.append(f"- {figures_dir / 'regime_return_boxplot.png'}")
    report_lines.append(f"- {figures_dir / 'run_length_hist.png'}")
    report_lines.append("")

    report_path = OUTPUT_DIR / "gmm_validation_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("Done.")
    print(f"Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

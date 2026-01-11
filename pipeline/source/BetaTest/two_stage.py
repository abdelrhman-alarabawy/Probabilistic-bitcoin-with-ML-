from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler


def _try_import(module_name: str):
    try:
        return __import__(module_name)
    except ImportError:
        return None


def _gate_model(seed: int):
    lgb = _try_import("lightgbm")
    if lgb:
        return "lightgbm", lgb.LGBMClassifier(
            objective="binary",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
        )
    return "logistic", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=seed,
        solver="liblinear",
    )


def _direction_model(seed: int):
    lgb = _try_import("lightgbm")
    if lgb:
        return "lightgbm", lgb.LGBMClassifier(
            objective="binary",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
        )
    return "logistic", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=seed,
        solver="liblinear",
    )


def sweep_gate_thresholds(
    p_trade: np.ndarray,
    y_true: np.ndarray,
    thresholds: List[float],
    min_coverage: float,
) -> Tuple[List[Dict[str, float]], Optional[Dict[str, float]]]:
    rows: List[Dict[str, float]] = []
    best = None
    for th in thresholds:
        preds = (p_trade >= th).astype(int)
        coverage = float(preds.mean())
        if coverage < min_coverage:
            continue
        precision = precision_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        row = {
            "gate_threshold": th,
            "precision_trade": float(precision),
            "recall_trade": float(recall),
            "f1_trade": float(f1),
            "coverage_trade": coverage,
        }
        rows.append(row)
        if best is None:
            best = row
            continue
        if row["precision_trade"] > best["precision_trade"]:
            best = row
        elif row["precision_trade"] == best["precision_trade"]:
            if row["coverage_trade"] > best["coverage_trade"]:
                best = row
            elif row["coverage_trade"] == best["coverage_trade"]:
                if row["recall_trade"] > best["recall_trade"]:
                    best = row

    if best is None:
        print(
            f"No feasible gate thresholds found for MIN_COVERAGE_TRADE={min_coverage}. "
            "Lower MIN_COVERAGE_TRADE or expand grid."
        )
    return rows, best


def sweep_direction_thresholds(
    p_long: np.ndarray,
    y_true: List[str],
    thresholds: List[float],
    min_dir_samples: int,
) -> Tuple[List[Dict[str, float]], Optional[Dict[str, float]]]:
    if len(y_true) < min_dir_samples:
        print(
            f"Not enough gated samples for direction tuning (n={len(y_true)}). "
            f"MIN_DIR_SAMPLES={min_dir_samples}."
        )
        return [], None

    rows: List[Dict[str, float]] = []
    best = None
    for th in thresholds:
        preds = ["long" if p >= th else "short" for p in p_long]
        precisions = precision_score(
            y_true,
            preds,
            labels=["long", "short"],
            average=None,
            zero_division=0,
        )
        precision_long = float(precisions[0])
        precision_short = float(precisions[1])
        score = (precision_long + precision_short) / 2
        row = {
            "dir_threshold": th,
            "precision_long": precision_long,
            "precision_short": precision_short,
            "score": float(score),
        }
        rows.append(row)
        if best is None:
            best = row
            continue
        if row["score"] > best["score"]:
            best = row
        elif row["score"] == best["score"]:
            if row["precision_long"] + row["precision_short"] > best["precision_long"] + best["precision_short"]:
                best = row

    return rows, best


def sweep_direction_band(
    p_long: np.ndarray,
    y_true: List[str],
    long_highs: List[float],
    short_lows: List[float],
    min_dir_samples: int,
    min_coverage_total: float,
    gate_coverage: float,
) -> Tuple[List[Dict[str, float]], Optional[Dict[str, float]]]:
    if len(y_true) < min_dir_samples:
        print(
            f"Not enough gated samples for direction tuning (n={len(y_true)}). "
            f"MIN_DIR_SAMPLES={min_dir_samples}."
        )
        return [], None

    rows: List[Dict[str, float]] = []
    best = None

    for high in long_highs:
        for low in short_lows:
            if low >= high:
                continue
            preds = []
            for p in p_long:
                if p >= high:
                    preds.append("long")
                elif p <= low:
                    preds.append("short")
                else:
                    preds.append("skip")

            pred_long = sum(1 for p in preds if p == "long")
            pred_short = sum(1 for p in preds if p == "short")
            pred_skip = sum(1 for p in preds if p == "skip")
            coverage_dir = (pred_long + pred_short) / len(preds) if preds else 0.0
            coverage_total = coverage_dir * gate_coverage
            if coverage_total < min_coverage_total:
                continue

            precisions = precision_score(
                y_true,
                preds,
                labels=["long", "short"],
                average=None,
                zero_division=0,
            )
            precision_long = float(precisions[0])
            precision_short = float(precisions[1])
            trade_precision = 0.0
            trade_count = pred_long + pred_short
            if trade_count > 0:
                trade_precision = (
                    sum(1 for idx, pred in enumerate(preds) if pred in ("long", "short") and pred == y_true[idx])
                    / trade_count
                )

            score = (precision_long + precision_short) / 2
            row = {
                "dir_long_high": float(high),
                "dir_short_low": float(low),
                "precision_long": precision_long,
                "precision_short": precision_short,
                "trade_precision": float(trade_precision),
                "coverage_total": float(coverage_total),
                "coverage_dir": float(coverage_dir),
                "pred_long": pred_long,
                "pred_short": pred_short,
                "pred_skip": pred_skip,
                "score": float(score),
            }
            rows.append(row)

            if best is None:
                best = row
                continue
            if row["score"] > best["score"]:
                best = row
            elif row["score"] == best["score"]:
                if row["trade_precision"] > best["trade_precision"]:
                    best = row
                elif row["trade_precision"] == best["trade_precision"]:
                    if row["coverage_total"] > best["coverage_total"]:
                        best = row

    if best is None:
        print(
            f"No feasible direction band found for MIN_COVERAGE_TOTAL={min_coverage_total}. "
            "Lower MIN_COVERAGE_TOTAL or expand grid."
        )
    return rows, best


def run_two_stage(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    gate_thresholds: List[float],
    dir_thresholds: List[float],
    dir_long_highs: List[float],
    dir_short_lows: List[float],
    min_coverage_trade: float,
    min_dir_samples: int,
    min_coverage_total: float,
    seed: int,
) -> Dict[str, object]:
    y_gate_train = np.isin(y_train, ["long", "short"]).astype(int)
    y_gate_test = np.isin(y_test, ["long", "short"]).astype(int)

    gate_name, gate_model = _gate_model(seed)
    if gate_name == "logistic":
        scaler = StandardScaler()
        X_train_gate = scaler.fit_transform(X_train)
        X_test_gate = scaler.transform(X_test)
    else:
        X_train_gate = X_train
        X_test_gate = X_test

    gate_model.fit(X_train_gate, y_gate_train)
    p_trade = gate_model.predict_proba(X_test_gate)[:, 1]

    gate_grid, gate_best = sweep_gate_thresholds(
        p_trade,
        y_gate_test,
        gate_thresholds,
        min_coverage_trade,
    )

    if gate_best is None:
        return {
            "gate_model": gate_name,
            "gate_best": None,
            "gate_grid": gate_grid,
            "direction_best": None,
            "direction_grid": [],
            "direction_band_best": None,
            "direction_band_grid": [],
            "end_to_end": None,
        }

    gate_threshold = gate_best["gate_threshold"]
    gate_preds = (p_trade >= gate_threshold)
    gated_idx = np.where(gate_preds)[0]
    gate_coverage = float(gate_preds.mean())

    trade_mask_train = np.isin(y_train, ["long", "short"])
    X_train_trade = X_train[trade_mask_train]
    y_train_trade = y_train[trade_mask_train]

    direction_results = {
        "direction_best": None,
        "direction_grid": [],
        "direction_band_best": None,
        "direction_band_grid": [],
        "gated_count": int(len(gated_idx)),
    }

    end_to_end = None
    p_long_gated = None
    y_test_gated = None

    if len(X_train_trade) > 0 and len(gated_idx) > 0:
        dir_name, dir_model = _direction_model(seed)
        y_dir_train = np.array([1 if label == "long" else 0 for label in y_train_trade])

        if dir_name == "logistic":
            scaler_dir = StandardScaler()
            X_train_dir = scaler_dir.fit_transform(X_train_trade)
            X_test_dir = scaler_dir.transform(X_test[gated_idx])
        else:
            X_train_dir = X_train_trade
            X_test_dir = X_test[gated_idx]

        dir_model.fit(X_train_dir, y_dir_train)
        p_long = dir_model.predict_proba(X_test_dir)[:, 1]
        y_test_gated = y_test[gated_idx].tolist()
        p_long_gated = p_long

        dir_grid, dir_best = sweep_direction_thresholds(
            p_long,
            y_test_gated,
            dir_thresholds,
            min_dir_samples,
        )
        direction_results["direction_best"] = dir_best
        direction_results["direction_grid"] = dir_grid

        band_grid, band_best = sweep_direction_band(
            p_long,
            y_test_gated,
            dir_long_highs,
            dir_short_lows,
            min_dir_samples,
            min_coverage_total,
            gate_coverage,
        )
        direction_results["direction_band_best"] = band_best
        direction_results["direction_band_grid"] = band_grid

        if band_best is not None:
            long_high = band_best["dir_long_high"]
            short_low = band_best["dir_short_low"]
            dir_preds = []
            for p in p_long:
                if p >= long_high:
                    dir_preds.append("long")
                elif p <= short_low:
                    dir_preds.append("short")
                else:
                    dir_preds.append("skip")
        elif dir_best is not None:
            dir_threshold = dir_best["dir_threshold"]
            dir_preds = ["long" if p >= dir_threshold else "short" for p in p_long]
        else:
            dir_preds = ["skip" for _ in p_long]

        end_preds = np.array(["skip"] * len(y_test), dtype=object)
        end_preds[gated_idx] = dir_preds

        precisions = precision_score(
            y_test.tolist(),
            end_preds.tolist(),
            labels=["long", "short", "skip"],
            average=None,
            zero_division=0,
        )
        precision_long = float(precisions[0])
        precision_short = float(precisions[1])
        coverage_total = float(np.isin(end_preds, ["long", "short"]).mean())

        end_to_end = {
            "precision_long": precision_long,
            "precision_short": precision_short,
            "coverage_total": coverage_total,
            "pred_long": int(np.sum(end_preds == "long")),
            "pred_short": int(np.sum(end_preds == "short")),
            "pred_skip": int(np.sum(end_preds == "skip")),
            "preds": end_preds,
            "gate_threshold": gate_threshold,
            "gate_model": gate_name,
            "dir_model": dir_name,
            "gate_coverage": gate_coverage,
        }

    return {
        "gate_model": gate_name,
        "gate_best": gate_best,
        "gate_grid": gate_grid,
        "direction_best": direction_results.get("direction_best"),
        "direction_grid": direction_results.get("direction_grid"),
        "direction_band_best": direction_results.get("direction_band_best"),
        "direction_band_grid": direction_results.get("direction_band_grid"),
        "gated_count": direction_results.get("gated_count"),
        "end_to_end": end_to_end,
        "p_trade": p_trade,
        "gate_preds": gate_preds,
        "p_long_gated": p_long_gated,
        "y_test_gated": y_test_gated,
        "gated_idx": gated_idx,
    }
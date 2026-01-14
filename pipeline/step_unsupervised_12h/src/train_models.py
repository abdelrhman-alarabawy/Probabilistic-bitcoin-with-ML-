from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.utils.class_weight import compute_class_weight


try:
    from lightgbm import LGBMClassifier  # type: ignore

    LGBM_AVAILABLE = True
except Exception:  # pragma: no cover
    LGBMClassifier = None
    LGBM_AVAILABLE = False

try:
    from xgboost import XGBClassifier  # type: ignore

    XGB_AVAILABLE = True
except Exception:  # pragma: no cover
    XGBClassifier = None
    XGB_AVAILABLE = False

try:
    from catboost import CatBoostClassifier  # type: ignore

    CAT_AVAILABLE = True
except Exception:  # pragma: no cover
    CatBoostClassifier = None
    CAT_AVAILABLE = False

from sklearn.ensemble import RandomForestClassifier


@dataclass
class ModelResult:
    model: object
    label_encoder: LabelEncoder
    metrics: dict
    confusion: np.ndarray
    feature_columns: list[str]


def chronological_split(df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15):
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, n)


def select_numeric_features(
    df: pd.DataFrame,
    exclude: list[str],
) -> list[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in exclude]
    feature_cols = [col for col in feature_cols if df[col].nunique(dropna=True) > 1]
    return feature_cols


def build_feature_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    train_idx: slice,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, SimpleImputer, RobustScaler]:
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    train_data = df.iloc[train_idx][feature_cols]
    imputer.fit(train_data)
    train_imputed = imputer.transform(train_data)
    scaler.fit(train_imputed)
    full_imputed = imputer.transform(df[feature_cols])
    full_scaled = scaler.transform(full_imputed)
    return (
        full_scaled,
        imputer.transform(train_data),
        scaler.transform(train_imputed),
        imputer,
        scaler,
    )


def pick_model(num_classes: int, random_state: int):
    if LGBM_AVAILABLE:
        return LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=random_state,
            class_weight="balanced",
        )
    if XGB_AVAILABLE:
        return XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=num_classes,
            random_state=random_state,
        )
    if CAT_AVAILABLE:
        return CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            loss_function="MultiClass",
            random_state=random_state,
            verbose=False,
        )
    return RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        class_weight="balanced",
    )


def train_classifier(
    df: pd.DataFrame,
    label_col: str,
    exclude_cols: list[str],
    output_dir: Path,
    random_state: int = 42,
) -> Optional[ModelResult]:
    df = df.copy()
    df = df.dropna(subset=[label_col])
    if len(df) < 200:
        return None

    train_idx, val_idx, test_idx = chronological_split(df)
    feature_cols = select_numeric_features(df, exclude_cols + [label_col])
    if not feature_cols:
        return None

    X_full, _, _, imputer, scaler = build_feature_matrix(df, feature_cols, train_idx)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[label_col].astype(str))
    if len(label_encoder.classes_) < 2:
        return None

    X_train = X_full[train_idx]
    y_train = y[train_idx]
    X_val = X_full[val_idx]
    y_val = y[val_idx]
    X_test = X_full[test_idx]
    y_test = y[test_idx]

    classes = np.unique(y_train)
    class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
    weights_map = {cls: w for cls, w in zip(classes, class_weights)}
    sample_weight = np.array([weights_map[label] for label in y_train])

    model = pick_model(len(label_encoder.classes_), random_state)
    if hasattr(model, "fit"):
        fit_kwargs = {}
        if model.__class__.__name__.startswith("LGBM"):
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["sample_weight"] = sample_weight
        elif model.__class__.__name__.startswith("XGB"):
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["sample_weight"] = sample_weight
        elif model.__class__.__name__.startswith("CatBoost"):
            fit_kwargs["sample_weight"] = sample_weight
        elif model.__class__.__name__.startswith("RandomForest"):
            fit_kwargs["sample_weight"] = sample_weight
        model.fit(X_train, y_train, **fit_kwargs)

    preds = model.predict(X_test)
    confusion = confusion_matrix(y_test, preds)
    acc = accuracy_score(y_test, preds)
    f1_macro = f1_score(y_test, preds, average="macro")
    f1_weighted = f1_score(y_test, preds, average="weighted")
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, preds, labels=classes, zero_division=0
    )
    metrics = {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "classes": label_encoder.classes_.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1_per_class": f1.tolist(),
        "support": support.tolist(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    report = classification_report(
        y_test,
        preds,
        target_names=label_encoder.classes_,
        zero_division=0,
    )
    report_path = output_dir / "classification_report.txt"
    report_path.write_text(report, encoding="utf-8")

    return ModelResult(
        model=model,
        label_encoder=label_encoder,
        metrics=metrics,
        confusion=confusion,
        feature_columns=feature_cols,
    )


def save_model(model_result: ModelResult, output_dir: Path) -> None:
    model = model_result.model
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.pkl"
    if hasattr(model, "save_model"):
        model.save_model(str(output_dir / "model.txt"))
    else:
        import joblib

        joblib.dump(model, model_path)

    encoder_path = output_dir / "label_encoder.pkl"
    try:
        import joblib

        joblib.dump(model_result.label_encoder, encoder_path)
    except Exception:
        pass


def save_confusion_plot(
    confusion: np.ndarray,
    labels: list[str],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(confusion, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, confusion[i, j], ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

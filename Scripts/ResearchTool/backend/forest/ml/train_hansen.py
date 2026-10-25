from __future__ import annotations

import json
import sys
from pathlib import Path

_here = Path(__file__).resolve()
for _p in _here.parents:
    if (_p / "backend").exists():
        sys.path.insert(0, str(_p))
        break

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from backend.config import FOREST_FEATURES_DIR, FOREST_MODELS_DIR


# === Конфиг ===
DATASET = FOREST_FEATURES_DIR / "training_features_valid.parquet"

# Финальный набор признаков (без координат и POST-влажности),
# который дает честный F1 = 0.862 без confounding.
FEATURE_COLS = [
    # Спектральные относительные дельты (PRE -> POST)
    "ndvi_median_rel",
    "nbr_median_rel",
    "ndmi_median_rel",

    # Контекст Hansen
    "hansen_loss_fraction",
    "hansen_recent_loss_5y",
    "hansen_dominant_share",
]

MIN_CLASS_SIZE = 20
N_SPLITS = 5
ONLY_DRIVERS: tuple[str, ...] | None = ("Wildfire", "Logging")

# "group" — GroupKFold по GROUP_COL (честная проверка по годам)
CV_MODE = "group"
GROUP_COL = "hansen_event_year"

# Имя для сохранения финальной модели и метрик
MODEL_NAME = "hansen_driver_final"
RANDOM_STATE = 42


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    print(f"Исходно: {len(df)}")

    df = df.dropna(subset=["hansen_dominant_share"])
    df = df[df["hansen_dominant_share"] > 0]
    print(f"После dominant_share > 0: {len(df)}")

    if ONLY_DRIVERS:
        before = len(df)
        df = df[df["driver_primary"].isin(ONLY_DRIVERS)].copy()
        print(f"Только {ONLY_DRIVERS}: {len(df)} (убрано {before - len(df)})")

    counts = df["driver_primary"].value_counts()
    rare = counts[counts < MIN_CLASS_SIZE].index.tolist()
    if rare:
        print(f"Редкие классы (< {MIN_CLASS_SIZE}) -> Other: {rare}")
        df = df.copy()
        df["driver_primary"] = df["driver_primary"].where(
            ~df["driver_primary"].isin(rare), other="Other"
        )

    print("\nКлассы:")
    print(df["driver_primary"].value_counts())
    return df.reset_index(drop=True)


def make_classifier(n_classes: int, **overrides) -> xgb.XGBClassifier:
    params = {
        "n_estimators": 400,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    params.update(overrides)

    if n_classes == 2:
        params["objective"] = "binary:logistic"
        params["eval_metric"] = "logloss"
    else:
        params["objective"] = "multi:softprob"
        params["num_class"] = n_classes
        params["eval_metric"] = "mlogloss"

    return xgb.XGBClassifier(**params)


def build_cv(df: pd.DataFrame, y: np.ndarray):
    """Возвращает (cv, groups_or_None, description)."""
    if CV_MODE == "group":
        if GROUP_COL not in df.columns:
            raise RuntimeError(f"Нет колонки {GROUP_COL} для GroupKFold")

        groups = df[GROUP_COL].fillna("unknown").astype(str).values
        n_groups = pd.Series(groups).nunique()

        if n_groups < 2:
            print(f"[!] {GROUP_COL} имеет только {n_groups} группу — fallback на StratifiedKFold")
            cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
            return cv, None, f"StratifiedKFold n_splits={N_SPLITS} (fallback)"

        n_splits = min(N_SPLITS, n_groups)
        desc = f"GroupKFold по {GROUP_COL}, n_splits={n_splits} (groups={n_groups})"
        return GroupKFold(n_splits=n_splits), groups, desc

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    return cv, None, f"StratifiedKFold n_splits={N_SPLITS}"


def main() -> None:
    if not DATASET.exists():
        raise FileNotFoundError(
            f"Нет {DATASET}. Сначала запустите build_training_dataset.py"
        )

    df = pd.read_parquet(DATASET)
    df = prepare(df)
    if df.empty:
        raise RuntimeError("Пустой датасет после фильтрации")

    feats = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"\n[!] нет в датасете, пропускаю: {missing}")
    print(f"\nИспользую признаков: {len(feats)}")

    X = df[feats].copy()
    medians = X.median(numeric_only=True)
    X = X.fillna(medians).fillna(0.0).reset_index(drop=True)

    le = LabelEncoder()
    y = le.fit_transform(df["driver_primary"])
    class_names = list(le.classes_)
    n_classes = len(class_names)

    print(f"Классов: {n_classes} -> {class_names}")
    print(f"Строк: {len(df)}")
    print(f"Распределение: {dict(zip(class_names, np.bincount(y).tolist()))}")

    cv, groups, cv_desc = build_cv(df, y)
    print(f"\nCV: {cv_desc}\n")

    oof = np.empty(len(df), dtype=int)
    oof_proba = np.zeros((len(df), n_classes))

    split_iter = cv.split(X, y, groups) if groups is not None else cv.split(X, y)
    for fold, (tr, va) in enumerate(split_iter):
        model = make_classifier(n_classes)
        model.fit(X.iloc[tr], y[tr])
        proba = model.predict_proba(X.iloc[va])
        oof[va] = np.argmax(proba, axis=1)
        oof_proba[va] = proba

        val_groups = (
            sorted(set(df[GROUP_COL].fillna("unknown").astype(str).iloc[va]))
            if GROUP_COL in df.columns else []
        )
        print(f"  fold {fold}: train={len(tr)} val={len(va)}  val_years={val_groups}")

    print()
    print("=" * 70)
    print(f"OOF METRICS ({cv_desc})")
    print("=" * 70)
    print(classification_report(
        y, oof,
        labels=range(n_classes),
        target_names=class_names,
        digits=3,
        zero_division=0,
    ))

    print("Confusion matrix:")
    cm = confusion_matrix(y, oof, labels=range(n_classes))
    header = " " * 28 + " ".join(f"{i:>6d}" for i in range(n_classes))
    print(header)
    for i, row in enumerate(cm):
        print(f"{class_names[i]:>26s} " + " ".join(f"{v:>6d}" for v in row))
    print()

    f1m = f1_score(y, oof, average="macro", zero_division=0)
    print(f"\nF1-macro: {f1m:.3f}")

    print("\nОбучаю финальную модель на всех данных...")
    final = make_classifier(n_classes)
    final.fit(X, y)

    FOREST_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = FOREST_MODELS_DIR / f"{MODEL_NAME}.json"
    final.save_model(model_path)
    print(f"[ok] модель сохранена: {model_path}")

    imp = pd.DataFrame({
        "feature": feats,
        "importance": final.feature_importances_,
    }).sort_values("importance", ascending=False)
    print()
    print("=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)
    print(imp.to_string(index=False))

    metrics = {
        "n_samples": len(df),
        "n_features": len(feats),
        "n_classes": int(n_classes),
        "class_names": class_names,
        "class_counts": dict(zip(class_names, np.bincount(y).tolist())),
        "f1_macro": float(f1m),
        "cv_mode": CV_MODE,
        "cv_description": cv_desc,
        "group_col": GROUP_COL,
        "top_features": imp.head(10).to_dict("records"),
        "features_used": feats,
        "dataset": str(DATASET),
    }
    metrics_path = FOREST_MODELS_DIR / f"{MODEL_NAME}_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[ok] метрики: {metrics_path}")

    oof_df = pd.DataFrame({
        "sample_id": df["sample_id"].values,
        GROUP_COL: df[GROUP_COL].values if GROUP_COL in df.columns else "",
        "region": df["region"].values if "region" in df.columns else "",
        "y_true": [class_names[i] for i in y],
        "y_pred": [class_names[i] for i in oof],
        **{f"proba_{name}": oof_proba[:, i] for i, name in enumerate(class_names)},
    })
    oof_path = FOREST_MODELS_DIR / f"{MODEL_NAME}_oof.csv"
    oof_df.to_csv(oof_path, index=False, encoding="utf-8-sig")
    print(f"[ok] OOF predictions: {oof_path}")


if __name__ == "__main__":
    main()
from __future__ import annotations

import json
import sys
from pathlib import Path

# Ищем ResearchTool и добавляем в sys.path
_here = Path(__file__).resolve()
for _parent in _here.parents:
    if (_parent / "backend").exists():
        sys.path.insert(0, str(_parent))
        break

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from backend.config import FOREST_FEATURES_DIR, FOREST_MODELS_DIR


# === Настройки ===
FEATURE_COLS = [
    # "lat",
    # "lon",
    "hansen_loss_fraction",
    "hansen_recent_loss_5y",
    "hansen_dominant_share",
]

MIN_CLASS_SIZE = 30
EXCLUDE_STATUSES = {"NO_LOSS"}
N_SPLITS = 3


def load_features() -> pd.DataFrame:
    parquet = FOREST_FEATURES_DIR / "hansen_features.parquet"
    csv = FOREST_FEATURES_DIR / "hansen_features.csv"

    if parquet.exists():
        print(f"Читаю parquet: {parquet}")
        return pd.read_parquet(parquet)
    if csv.exists():
        print(f"Читаю CSV: {csv}")
        return pd.read_csv(csv, encoding="utf-8-sig")
    raise FileNotFoundError(
        f"Не найден ни {parquet}, ни {csv}. "
        f"Сначала запустите export_hansen_features.py"
    )


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    print(f"Исходно: {len(df)} строк")

    df = df.dropna(subset=["hansen_dominant_share"])
    print(f"После dropna(dominant_share): {len(df)}")

    df = df[df["hansen_dominant_share"] > 0]
    print(f"После dominant_share > 0: {len(df)}")

    if "hansen_status" in df.columns:
        before = len(df)
        df = df[~df["hansen_status"].isin(EXCLUDE_STATUSES)]
        print(f"После исключения {EXCLUDE_STATUSES}: {len(df)} (убрано {before - len(df)})")

    counts = df["driver_primary"].value_counts()
    rare = counts[counts < MIN_CLASS_SIZE].index.tolist()
    if rare:
        print(f"Редкие классы (< {MIN_CLASS_SIZE}): {rare}")
        df = df.copy()
        df["driver_primary"] = df["driver_primary"].where(
            ~df["driver_primary"].isin(rare), other="Other"
        )

    print()
    print("Классы для обучения:")
    print(df["driver_primary"].value_counts())
    print()

    #Test 2 - without lat, lon and two class
    df = df[df["driver_primary"].isin(["Wildfire", "Logging"])]
    print(f"Оставлено только Wildfire/Logging: {len(df)}")

    return df


def make_classifier(n_classes: int, **overrides) -> xgb.XGBClassifier:
    params = {
        "n_estimators": 400,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "random_state": 42,
    }
    params.update(overrides)

    if n_classes == 2:
        params["objective"] = "binary:logistic"
    else:
        params["objective"] = "multi:softprob"
        params["num_class"] = n_classes

    return xgb.XGBClassifier(**params)


def train() -> dict:
    df = load_features()
    df = prepare_dataset(df)

    if df.empty:
        raise RuntimeError("Пустой датасет после фильтрации")

    X = df[FEATURE_COLS].fillna(0.0).reset_index(drop=True)
    groups = df["region"].fillna("unknown").reset_index(drop=True)

    le = LabelEncoder()
    y = le.fit_transform(df["driver_primary"])
    class_names = list(le.classes_)
    n_classes = len(class_names)

    print(f"Признаков: {len(FEATURE_COLS)}")
    print(f"Классов: {n_classes}")
    print(f"Классы (по индексу):")
    for idx, name in enumerate(class_names):
        print(f"  {idx}: {name}")
    print(f"Регионов: {groups.nunique()}")
    print()

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    oof = np.empty(len(df), dtype=int)
    oof_proba = np.zeros((len(df), n_classes))

    for fold, (tr, va) in enumerate(cv.split(X, y)):
        model = make_classifier(n_classes)
        model.fit(X.iloc[tr], y[tr])

        proba = model.predict_proba(X.iloc[va])
        oof[va] = np.argmax(proba, axis=1)
        oof_proba[va] = proba

        print(f"  fold {fold}: обучено (train={len(tr)}, val={len(va)})")

    print()
    print("=" * 70)
    print("МЕТРИКИ НА КРОСС-ВАЛИДАЦИИ (OOF)")
    print("=" * 70)
    print()
    print(classification_report(
        y, oof,
        labels=range(n_classes),
        target_names=class_names,
        digits=3,
        zero_division=0,
    ))

    print("Confusion matrix:")
    cm = confusion_matrix(y, oof, labels=range(n_classes))
    header = " " * 32 + " ".join(f"{i:>6d}" for i in range(n_classes))
    print(header)
    for i, row in enumerate(cm):
        row_str = " ".join(f"{v:>6d}" for v in row)
        print(f"{class_names[i]:>30s} {row_str}")
    print()
    print("Порядок классов (индекс → имя):")
    for idx, name in enumerate(class_names):
        print(f"  {idx}: {name}")

    f1m = f1_score(y, oof, average="macro", zero_division=0)
    print(f"\nF1-macro: {f1m:.3f}")

    print()
    print("Обучаю финальную модель на всех данных...")
    final = make_classifier(n_classes)
    final.fit(X, y)

    FOREST_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = FOREST_MODELS_DIR / "hansen_driver_w_lat_lon.json"
    final.save_model(model_path)
    print(f"[ok] модель сохранена: {model_path}")

    imp = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": final.feature_importances_,
    }).sort_values("importance", ascending=False)
    print()
    print("=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)
    print(imp.to_string(index=False))

    metrics = {
        "n_samples": len(df),
        "n_features": len(FEATURE_COLS),
        "n_classes": int(n_classes),
        "class_names": class_names,
        "class_counts": dict(zip(class_names, np.bincount(y).tolist())),
        "f1_macro": float(f1m),
        "top_features": imp.head(5).to_dict("records"),
        "cv_splits": N_SPLITS,
        "min_class_size": MIN_CLASS_SIZE,
        "features_used": list(FEATURE_COLS),
    }
    metrics_path = FOREST_MODELS_DIR / "hansen_metrics_w_lat_lon.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[ok] метрики: {metrics_path}")

    return metrics


if __name__ == "__main__":
    train()
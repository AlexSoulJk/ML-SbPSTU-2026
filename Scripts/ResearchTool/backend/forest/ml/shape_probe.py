from __future__ import annotations

import sys
from pathlib import Path

_here = Path(__file__).resolve()
for _p in _here.parents:
    if (_p / "backend").exists():
        sys.path.insert(0, str(_p))
        break

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

from backend.forest.ml.train_hansen import (
    DATASET,
    N_SPLITS,
    make_classifier,
    prepare,
)
from backend.config import FOREST_MODELS_DIR


GROUP_COL = "hansen_event_year"

# Полный список shape-признаков (из group_ablation)
SHAPE_ALL = [
    "loss_n_components",
    "loss_largest_share",
    "loss_mean_area_ha",
    "loss_compactness",
    "loss_rectangularity",
    "loss_perimeter_area_ratio",
    "loss_edge_density",
]

# Наборы для теста 1: от самого полного к минимальному
SHAPE_SUBSETS = {
    "SHAPE_7 (все)": SHAPE_ALL,
    "SHAPE_5": [
        "loss_n_components",
        "loss_largest_share",
        "loss_mean_area_ha",
        "loss_compactness",
        "loss_rectangularity",
    ],
    "SHAPE_3 (форма)": [
        "loss_n_components",
        "loss_largest_share",
        "loss_compactness",
    ],
    "SHAPE_2 (размер + форма)": [
        "loss_n_components",
        "loss_compactness",
    ],
    "SHAPE_2 (крупность)": [
        "loss_n_components",
        "loss_largest_share",
    ],
    "SHAPE_2 (прямоугольность)": [
        "loss_rectangularity",
        "loss_compactness",
    ],
}

# Наборы для теста 2: чистый сигнал без confounding
TEST2_COMBOS = {
    "rel + SHAPE3 + HANSEN": [
        "ndmi_median_rel", "nbr_median_rel", "ndvi_median_rel",
        "loss_n_components", "loss_largest_share", "loss_compactness",
        "hansen_loss_fraction", "hansen_recent_loss_5y", "hansen_dominant_share",
    ],
    "rel + SHAPE2 + HANSEN": [
        "ndmi_median_rel", "nbr_median_rel", "ndvi_median_rel",
        "loss_n_components", "loss_compactness",
        "hansen_loss_fraction", "hansen_recent_loss_5y", "hansen_dominant_share",
    ],
    "rel + HANSEN": [
        "ndmi_median_rel", "nbr_median_rel", "ndvi_median_rel",
        "hansen_loss_fraction", "hansen_recent_loss_5y", "hansen_dominant_share",
    ],
}

# Наборы для теста 3: shape vs spatial
TEST3_COMBOS = {
    "SPATIAL": ["lat", "lon"],
    "SHAPE_3": ["loss_n_components", "loss_largest_share", "loss_compactness"],
    "SHAPE_3 + SPATIAL": [
        "loss_n_components", "loss_largest_share", "loss_compactness",
        "lat", "lon",
    ],
    "SHAPE_3 + SPATIAL + HANSEN": [
        "loss_n_components", "loss_largest_share", "loss_compactness",
        "lat", "lon",
        "hansen_loss_fraction", "hansen_recent_loss_5y", "hansen_dominant_share",
    ],
}


def train_features(df, y, feats, n_classes, groups):
    """OOF F1-macro для заданного набора признаков."""
    feats = [c for c in feats if c in df.columns]
    if not feats:
        return None, 0

    X = df[feats].copy()
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0).reset_index(drop=True)

    cv = GroupKFold(n_splits=min(N_SPLITS, pd.Series(groups).nunique()))
    oof = np.empty(len(df), dtype=int)
    for tr, va in cv.split(X, y, groups):
        model = make_classifier(n_classes)
        model.fit(X.iloc[tr], y[tr])
        oof[va] = np.argmax(model.predict_proba(X.iloc[va]), axis=1)

    return f1_score(y, oof, average="macro", zero_division=0), len(feats)


def run_block(title: str, combos: dict, df, y, n_classes, groups) -> list[dict]:
    print(f"\n=== {title} ===")
    rows = []
    for name, cols in combos.items():
        missing = [c for c in cols if c not in df.columns]
        if missing:
            print(f"  [!] {name}: нет в датасете: {missing}")
        f1, n = train_features(df, y, cols, n_classes, groups)
        if f1 is None:
            continue
        rows.append({"block": title, "name": name, "n_features": n, "f1_macro": round(f1, 3)})
        print(f"  {name:<28s}  ({n} признаков)  F1-macro = {f1:.3f}")
    return rows


def main() -> None:
    df = pd.read_parquet(DATASET)
    df = prepare(df)
    if df.empty:
        raise RuntimeError("Пустой датасет")

    le = LabelEncoder()
    y = le.fit_transform(df["driver_primary"])
    n_classes = len(le.classes_)
    groups = df[GROUP_COL].fillna("unknown").astype(str).values

    baseline = f1_score(
        y, np.full_like(y, np.bincount(y).argmax()),
        average="macro", zero_division=0,
    )
    print(f"Строк: {len(df)}, классов: {n_classes}, групп: {pd.Series(groups).nunique()}")
    print(f"Baseline (мажоритарный класс): F1-macro = {baseline:.3f}")

    all_rows = []

    # Тест 1: сколько признаков формы нужно
    all_rows += run_block(
        "Тест 1. Сколько shape-признаков нужно",
        SHAPE_SUBSETS, df, y, n_classes, groups,
    )

    # Тест 2: чистая модель без confounding
    all_rows += run_block(
        "Тест 2. Чистый сигнал без координат и POST",
        TEST2_COMBOS, df, y, n_classes, groups,
    )

    # Тест 3: shape vs spatial
    all_rows += run_block(
        "Тест 3. Форма vs география",
        TEST3_COMBOS, df, y, n_classes, groups,
    )

    out = pd.DataFrame(all_rows)
    print("\n=== СВОДНАЯ ТАБЛИЦА ===")
    print(out.sort_values(["block", "f1_macro"], ascending=[True, False]).to_string(index=False))

    FOREST_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = FOREST_MODELS_DIR / "shape_probe.csv"
    out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n[ok] {csv_path}")


if __name__ == "__main__":
    main()
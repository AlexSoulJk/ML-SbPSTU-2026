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

GROUPS = {
    "CHG (все 10)": [
        "chg_matched_area_ha", "chg_max_matched_area_ha", "chg_matched_dnbr_mean",
        "chg_n_matched", "chg_matched_overlap_mean", "chg_matched_compactness_mean",
        "chg_max_matched_compactness", "chg_matched_rectangularity_mean",
        "chg_max_matched_rectangularity", "chg_matched_solidity_mean",
    ],
    "CHG (только форма)": [
        "chg_matched_compactness_mean", "chg_matched_rectangularity_mean", "chg_matched_solidity_mean",
    ],
    "CHG (только площадь)": [
        "chg_matched_area_ha", "chg_max_matched_area_ha", "chg_n_matched",
    ],
    "BASE (rel + Hansen)": [
        "ndmi_median_rel", "nbr_median_rel", "ndvi_median_rel",
        "hansen_loss_fraction", "hansen_recent_loss_5y", "hansen_dominant_share",
    ],
    "BASE + CHG (только форма)": [
        "ndmi_median_rel", "nbr_median_rel", "ndvi_median_rel",
        "hansen_loss_fraction", "hansen_recent_loss_5y", "hansen_dominant_share",
        "chg_matched_compactness_mean", "chg_matched_rectangularity_mean", "chg_matched_solidity_mean",
    ],
    "BASE + CHG (всё вместе)": [
        "ndmi_median_rel", "nbr_median_rel", "ndvi_median_rel",
        "hansen_loss_fraction", "hansen_recent_loss_5y", "hansen_dominant_share",
        "chg_matched_area_ha", "chg_max_matched_area_ha", "chg_matched_dnbr_mean",
        "chg_n_matched", "chg_matched_compactness_mean", "chg_matched_rectangularity_mean",
    ],
    "rel + chg (без Hansen)": [
        "ndmi_median_rel", "nbr_median_rel", "ndvi_median_rel",
        "chg_matched_area_ha", "chg_max_matched_area_ha", "chg_matched_dnbr_mean",
        "chg_n_matched", "chg_matched_compactness_mean", "chg_matched_rectangularity_mean",
    ],
    "rel + chg (только площадь)": [
        "ndmi_median_rel", "nbr_median_rel", "ndvi_median_rel",
        "chg_matched_area_ha", "chg_max_matched_area_ha", "chg_n_matched",
    ],
    "rel + Hansen (наш текущий BASE)": [
        "ndmi_median_rel", "nbr_median_rel", "ndvi_median_rel",
        "hansen_loss_fraction", "hansen_recent_loss_5y", "hansen_dominant_share",
    ],
    # --- НОВАЯ ГРУППА ---
    "BASE + ЛУЧШАЯ форма dNBR": [
        # Спектральная база (6)
        "ndmi_median_rel", "nbr_median_rel", "ndvi_median_rel",
        "hansen_loss_fraction", "hansen_recent_loss_5y", "hansen_dominant_share",
        # Лучшие 5 признаков формы dNBR (из теста на 0.690)
        "chg_n_matched",
        "chg_matched_area_ha",
        "chg_matched_compactness_mean",
        "chg_matched_rectangularity_mean",
        "chg_matched_solidity_mean",
    ]
}

SINGLE_FEATURES = [
    "chg_matched_compactness_mean",
    "chg_max_matched_compactness",
    "chg_matched_rectangularity_mean",
    "chg_max_matched_rectangularity",
    "chg_matched_solidity_mean",
]


def train_features(df, y, feats, n_classes, groups):
    feats = [c for c in feats if c in df.columns]
    if not feats: return None, 0

    X = df[feats].copy()
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0).reset_index(drop=True)

    cv = GroupKFold(n_splits=min(N_SPLITS, pd.Series(groups).nunique()))
    oof = np.empty(len(df), dtype=int)

    for tr, va in cv.split(X, y, groups):
        model = make_classifier(n_classes)
        model.fit(X.iloc[tr], y[tr])
        oof[va] = np.argmax(model.predict_proba(X.iloc[va]), axis=1)

    return f1_score(y, oof, average="macro", zero_division=0), len(feats)


def main() -> None:
    df = pd.read_parquet(DATASET)
    df = prepare(df)
    if df.empty: raise RuntimeError("Пустой датасет")

    le = LabelEncoder()
    y = le.fit_transform(df["driver_primary"])
    n_classes = len(le.classes_)
    groups = df[GROUP_COL].fillna("unknown").astype(str).values

    baseline = f1_score(y, np.full_like(y, np.bincount(y).argmax()), average="macro", zero_division=0)
    print(f"Строк: {len(df)}, классов: {n_classes}, групп: {pd.Series(groups).nunique()}")
    print(f"Baseline (мажоритарный класс): F1-macro = {baseline:.3f}\n")

    rows = []

    print("=== Одиночные признаки формы CHG ===")
    for feat in SINGLE_FEATURES:
        f1, n = train_features(df, y, [feat], n_classes, groups)
        if f1 is not None:
            rows.append({"group": "single", "name": feat, "n_features": n, "f1_macro": round(f1, 3)})
            print(f"  {feat:<32s}  F1-macro = {f1:.3f}")

    print("\n=== Группы и комбинации ===")
    for name, cols in GROUPS.items():
        f1, n = train_features(df, y, cols, n_classes, groups)
        if f1 is not None:
            rows.append({"group": "combo", "name": name, "n_features": n, "f1_macro": round(f1, 3)})
            print(f"  {name:<32s}  ({n} призн.)  F1-macro = {f1:.3f}")

    out = pd.DataFrame(rows)
    print("\n=== СВОДНАЯ ТАБЛИЦА ===")
    print(out.sort_values("f1_macro", ascending=False).to_string(index=False))

    FOREST_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(FOREST_MODELS_DIR / "change_ablation.csv", index=False, encoding="utf-8-sig")
    print(f"\n[ok] {FOREST_MODELS_DIR / 'change_ablation.csv'}")


if __name__ == "__main__":
    main()
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

# === Группы признаков ===
GROUPS: dict[str, list[str]] = {
    "PRE": [
        "ndmi_median_pre",
        "ndvi_median_pre",
        "nbr_median_pre",
        "scl_4_frac_pre",
        "scl_5_frac_pre",
    ],
    "POST": [
        "ndmi_median_post",
        "ndvi_median_post",
        "nbr_median_post",
        "scl_4_frac_post",
        "scl_5_frac_post",
    ],
    "DELTA": [
        "ndmi_median_delta",
        "ndvi_median_delta",
        "nbr_median_delta",
        "ndmi_median_rel",
        "ndvi_median_rel",
        "nbr_median_rel",
    ],
    "SPATIAL": [
        "lat",
        "lon",
    ],
    "HANSEN": [
        "hansen_loss_fraction",
        "hansen_recent_loss_5y",
        "hansen_dominant_share",
    ],
    "SHAPE": [
        "loss_n_components",
        "loss_largest_share",
        "loss_mean_area_ha",
        "loss_compactness",
        "loss_rectangularity",
        "loss_perimeter_area_ratio",
        "loss_edge_density",
    ],
}

GROUP_COL = "hansen_event_year"


def train_features(df, y, feats, n_classes, groups):
    """OOF F1-macro для заданного набора признаков."""
    X = df[feats].copy()
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0).reset_index(drop=True)

    cv = GroupKFold(n_splits=min(N_SPLITS, pd.Series(groups).nunique()))
    oof = np.empty(len(df), dtype=int)
    for tr, va in cv.split(X, y, groups):
        model = make_classifier(n_classes)
        model.fit(X.iloc[tr], y[tr])
        oof[va] = np.argmax(model.predict_proba(X.iloc[va]), axis=1)

    return f1_score(y, oof, average="macro", zero_division=0)


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
    print(f"Baseline (мажоритарный класс): F1-macro = {baseline:.3f}\n")

    rows = []
    for group_name, candidates in GROUPS.items():
        feats = [c for c in candidates if c in df.columns]
        missing = [c for c in candidates if c not in df.columns]
        if missing:
            print(f"[!] {group_name}: нет в датасете: {missing}")

        if not feats:
            continue

        f1 = train_features(df, y, feats, n_classes, groups)
        rows.append({
            "group": group_name,
            "n_features": len(feats),
            "features": ", ".join(feats),
            "f1_macro": round(f1, 3),
        })
        print(f"  {group_name:<10s}  ({len(feats)} признаков)  F1-macro = {f1:.3f}")

    # Комбинации: чистый сигнал + что-то ещё
    print("\n=== Комбинации ===")
    combos = {
        "DELTA + SHAPE":           ["ndmi_median_rel", "nbr_median_rel", "ndvi_median_rel"] + GROUPS["SHAPE"],
        "DELTA + SHAPE + HANSEN":  ["ndmi_median_rel", "nbr_median_rel", "ndvi_median_rel"] + GROUPS["SHAPE"] + GROUPS["HANSEN"],
        "SHAPE + HANSEN":          GROUPS["SHAPE"] + GROUPS["HANSEN"],
        "DELTA только rel":        ["ndmi_median_rel", "nbr_median_rel", "ndvi_median_rel"],
    }
    for name, cols in combos.items():
        feats = [c for c in cols if c in df.columns]
        if not feats:
            continue
        f1 = train_features(df, y, feats, n_classes, groups)
        rows.append({
            "group": name,
            "n_features": len(feats),
            "features": ", ".join(feats),
            "f1_macro": round(f1, 3),
        })
        print(f"  {name:<24s}  ({len(feats)} признаков)  F1-macro = {f1:.3f}")

    out = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    print("\n=== СВОДНАЯ ТАБЛИЦА ===")
    print(out[["group", "n_features", "f1_macro"]].to_string(index=False))

    from backend.config import FOREST_MODELS_DIR
    FOREST_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = FOREST_MODELS_DIR / "group_ablation.csv"
    out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n[ok] {csv_path}")


if __name__ == "__main__":
    main()
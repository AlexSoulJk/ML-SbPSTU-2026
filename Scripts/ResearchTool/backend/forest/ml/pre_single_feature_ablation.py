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

# === Группы кандидатов ===
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
}

GROUP_COL = "hansen_event_year"


def train_features(df, y, feats, n_classes, groups):
    """OOF F1 для произвольного набора признаков."""
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
    print(f"Baseline (majority): F1-macro = {baseline:.3f}\n")

    all_rows = []
    for group_name, candidates in GROUPS.items():
        feats = [c for c in candidates if c in df.columns]
        missing = [c for c in candidates if c not in df.columns]
        if missing:
            print(f"[!] {group_name}: нет в датасете: {missing}")

        if not feats:
            continue

        print(f"=== {group_name} ===")

        # 1) каждый признак в одиночку
        for feat in feats:
            f1 = train_features(df, y, [feat], n_classes, groups)
            print(f"  {feat:<24s}  F1 = {f1:.3f}")
            all_rows.append({"group": group_name, "features": feat, "f1_macro": round(f1, 3)})

        # 2) группа целиком
        f1_all = train_features(df, y, feats, n_classes, groups)
        print(f"  {'ВСЕ ' + group_name:<24s}  F1 = {f1_all:.3f}")
        all_rows.append({"group": group_name, "features": f"ALL({len(feats)})", "f1_macro": round(f1_all, 3)})
        print()

    out = pd.DataFrame(all_rows)
    print("=== СВОДНАЯ ТАБЛИЦА ===")
    print(out.sort_values("f1_macro", ascending=False).to_string(index=False))

    # Сохранить для отчёта
    from backend.config import FOREST_MODELS_DIR
    FOREST_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = FOREST_MODELS_DIR / "single_feature_ablation.csv"
    out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n[ok] {csv_path}")


if __name__ == "__main__":
    main()
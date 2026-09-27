from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

_here = Path(__file__).resolve()
for _p in _here.parents:
    if (_p / "backend").exists():
        sys.path.insert(0, str(_p))
        break

import numpy as np
import pandas as pd
import rasterio
from sqlalchemy import select, text

from backend.config import FOREST_FEATURES_DIR
from backend.forest.database import SessionLocal, init_forest_database
from backend.forest.models import (
    HansenAnalysis,
    Sample,
    SentinelDownload,
)
from backend.forest.storage_paths import resolve_storage_path


# === КОНФИГ ===
VALID_LABEL = "Valid"
PERIODS = ("PRE", "POST")
OUT_PARQUET = FOREST_FEATURES_DIR / "training_features_valid.parquet"

# Порядок бэндов жёстко из evalscript (см. request в sidecar JSON)
BAND_ORDER = ["B02", "B03", "B04", "B08", "B11", "B12", "SCL", "dataMask"]

# Индексы: имя -> (числитель, знаменатель)
INDEX_DEFS = {
    "ndvi": ("B08", "B04"),
    "nbr":  ("B08", "B12"),
    "ndmi": ("B08", "B11"),
}

# SCL-классы, которые считаем «плохими» (исключаем из статистик индексов)
SCL_BAD = {0, 1, 3, 8, 9, 10, 11}  # no-data, saturated, cloud_shadow, clouds, cirrus, snow

# SCL-классы, доли которых сохраняем как фичи
SCL_REPORT = (2, 4, 5, 6, 7)


# === 1. Кто Valid ===
def load_valid_ids(db) -> set[str]:
    rows = db.execute(
        text(
            "SELECT DISTINCT sample_id FROM forest_manual_validations "
            "WHERE validation = :v"
        ),
        {"v": VALID_LABEL},
    ).all()
    return {r[0] for r in rows}


# === 2. TIFF -> спектральные статистики одной сцены ===
def stats_for_tif(tif_path: Path) -> dict[str, float]:
    if not tif_path.exists():
        return {}

    with rasterio.open(tif_path) as src:
        if src.count != len(BAND_ORDER):
            return {}
        data = {
            name: src.read(i + 1).astype("float32")
            for i, name in enumerate(BAND_ORDER)
        }

    scl = data["SCL"].astype("int32")
    dm = data["dataMask"]

    # маска валидных пикселей: dataMask>0 и SCL не «плохой»
    bad_scl = np.isin(scl, list(SCL_BAD))
    valid = (dm > 0) & ~bad_scl

    out: dict[str, float] = {}

    # Индексы
    for name, (a_band, b_band) in INDEX_DEFS.items():
        a = data[a_band]
        b = data[b_band]
        denom = a + b
        ok = valid & np.isfinite(a) & np.isfinite(b) & (denom != 0)
        if not ok.any():
            continue
        idx = (a[ok] - b[ok]) / denom[ok]
        out[f"{name}_mean"] = float(idx.mean())
        out[f"{name}_median"] = float(np.median(idx))
        out[f"{name}_std"] = float(idx.std())
        out[f"{name}_p10"] = float(np.percentile(idx, 10))
        out[f"{name}_p90"] = float(np.percentile(idx, 90))
        out[f"{name}_valid_frac"] = float(ok.mean())

    # SCL-доли (по всей сцене, не по маске)
    total = scl.size
    for cls in SCL_REPORT:
        out[f"scl_{cls}_frac"] = float((scl == cls).sum() / total)

    return out


# === 3. Hansen-часть (только Valid, по одной записи на sample_id) ===
def collect_hansen(db, valid_ids: set[str]) -> pd.DataFrame:
    samples = {
        s.sample_id: s
        for s in db.scalars(select(Sample))
        if s.sample_id in valid_ids
    }

    # Последняя HansenAnalysis на sample_id (по created_at), как в export_hansen_features.py
    latest: dict[str, HansenAnalysis] = {}
    for a in db.scalars(select(HansenAnalysis).order_by(HansenAnalysis.created_at)):
        if a.sample_id in valid_ids:
            latest[a.sample_id] = a

    rows = []
    for sid, a in latest.items():
        s = samples.get(sid)
        if s is None:
            continue

        hist = json.loads(a.histogram_json or "[]")
        plot_area_ha = (a.plot_size_m ** 2) / 10_000.0 if a.plot_size_m else 0.0
        loss_fraction = (
            (a.total_loss_area_ha or 0.0) / plot_area_ha
            if plot_area_ha > 0 else 0.0
        )
        max_year = max((h["year"] for h in hist), default=0)
        cutoff = max_year - 5
        recent_ha = sum(h["loss_area_ha"] for h in hist if h["year"] > cutoff)
        recent_fraction = recent_ha / plot_area_ha if plot_area_ha > 0 else 0.0

        event_year = getattr(a, "event_year", None) or getattr(a, "even_year", None)

        rows.append({
            "sample_id": s.sample_id,
            "lat": s.lat,
            "lon": s.lon,
            "driver_primary": s.driver_primary,
            "region": s.region,
            "source_file": s.source_file,
            "hansen_event_year": event_year,
            "hansen_total_loss_ha": a.total_loss_area_ha,
            "hansen_dominant_share": a.dominant_year_share,
            "hansen_loss_fraction": round(loss_fraction, 4),
            "hansen_recent_loss_5y": round(recent_fraction, 4),
            "hansen_status": a.status,
        })

    return pd.DataFrame(rows)


# === 4. Спектр: TIFF -> агрегат по (sample_id, period) ===
def collect_spectral(db, valid_ids: set[str]) -> pd.DataFrame:
    buckets: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    n_ok = n_fail = n_missing = 0

    for d in db.scalars(select(SentinelDownload)):
        if d.sample_id not in valid_ids:
            continue
        period = (d.period or "").upper()
        if period not in PERIODS:
            continue
        tif_path = resolve_storage_path(d.local_path)
        if not tif_path.exists():
            n_missing += 1
            continue
        stats = stats_for_tif(tif_path)
        if not stats:
            n_fail += 1
            continue
        buckets[(d.sample_id, period)].append(stats)
        n_ok += 1

    print(f"  TIFF: ok={n_ok}, fail={n_fail}, missing={n_missing}")

    all_keys: set[str] = set()
    for items in buckets.values():
        for it in items:
            all_keys.update(it.keys())

    rows: dict[str, dict[str, float]] = defaultdict(dict)
    for (sid, period), items in buckets.items():
        for k in all_keys:
            vals = [it[k] for it in items if k in it]
            if vals:
                rows[sid][f"{k}_{period.lower()}"] = float(np.median(vals))

    df = pd.DataFrame.from_dict(rows, orient="index").reset_index(names="sample_id")

    # Дельты PRE -> POST
    delta = {}
    for col in list(df.columns):
        if not col.endswith("_pre"):
            continue
        base = col[:-4]
        post = f"{base}_post"
        if post in df.columns:
            delta[f"{base}_delta"] = df[post] - df[col]
            delta[f"{base}_rel"] = (df[post] - df[col]) / (df[col].abs() + 1e-6)
    if delta:
        df = pd.concat([df, pd.DataFrame(delta, index=df.index)], axis=1)

    return df


def main() -> None:
    init_forest_database()
    FOREST_FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    with SessionLocal() as db:
        valid_ids = load_valid_ids(db)
        print(f"Valid samples: {len(valid_ids)}")
        if not valid_ids:
            print("Нет Valid-сэмплов, нечего собирать.")
            return

        hansen = collect_hansen(db, valid_ids)
        print(f"Hansen rows:   {len(hansen)}")

        spectral = collect_spectral(db, valid_ids)
        print(f"Spectral rows: {len(spectral)}, cols: {len(spectral.columns) - 1}")

    df = hansen.merge(spectral, on="sample_id", how="left")
    print(f"Merged: {len(df)} rows, {len(df.columns)} cols")

    # Покрытие спектральных колонок
    spec_cols = [
        c for c in df.columns
        if c != "sample_id"
        and not c.startswith(("hansen_", "lat", "lon", "driver_", "region", "source_"))
    ]
    if spec_cols:
        miss = df[spec_cols].isna().mean().sort_values(ascending=False)
        print("\nSpectral coverage (доля NaN, топ-15):")
        print(miss.head(15).to_string())

    # Краткая сводка по ключевым фичам, чтобы глазами проверить порядок бэндов
    key_cols = [c for c in ("ndvi_median_pre", "ndvi_median_post", "ndvi_delta",
                            "nbr_median_pre", "nbr_median_post", "nbr_delta",
                            "ndmi_median_pre", "ndmi_median_post", "ndmi_delta")
                if c in df.columns]
    if key_cols:
        print("\nКлючевые спектральные фичи (describe):")
        print(df[key_cols].describe().T.to_string())

    df.to_parquet(OUT_PARQUET, index=False)
    print(f"\n[ok] {OUT_PARQUET} — {len(df)} строк")


if __name__ == "__main__":
    main()
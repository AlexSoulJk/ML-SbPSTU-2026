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

from backend.config import FOREST_FEATURES_DIR, FOREST_CHANGE_CACHE_DIR
from backend.forest.database import SessionLocal, init_forest_database
from backend.forest.models import (
    HansenAnalysis,
    Sample,
    SentinelDownload,
    HansenMask,
)
from backend.forest.storage_paths import resolve_storage_path
from backend.forest.ml.hansen_shape_features import loss_shape_features

# === КОНФИГ ===
VALID_LABEL = "Valid"
PERIODS = ("PRE", "POST")
OUT_PARQUET = FOREST_FEATURES_DIR / "training_features_valid.parquet"

BAND_ORDER = ["B02", "B03", "B04", "B08", "B11", "B12", "SCL", "dataMask"]
INDEX_DEFS = {"ndvi": ("B08", "B04"), "nbr": ("B08", "B12"), "ndmi": ("B08", "B11")}
SCL_BAD = {0, 1, 3, 8, 9, 10, 11}
SCL_REPORT = (2, 4, 5, 6, 7)


def load_valid_ids(db) -> set[str]:
    rows = db.execute(text("SELECT DISTINCT sample_id FROM forest_manual_validations WHERE validation = :v"),
                      {"v": VALID_LABEL}).all()
    return {r[0] for r in rows}


def stats_for_tif(tif_path: Path) -> dict[str, float]:
    if not tif_path.exists(): return {}
    with rasterio.open(tif_path) as src:
        if src.count != len(BAND_ORDER): return {}
        data = {name: src.read(i + 1).astype("float32") for i, name in enumerate(BAND_ORDER)}
    scl = data["SCL"].astype("int32")
    dm = data["dataMask"]
    bad_scl = np.isin(scl, list(SCL_BAD))
    valid = (dm > 0) & ~bad_scl
    out = {}
    for name, (a_band, b_band) in INDEX_DEFS.items():
        a, b = data[a_band], data[b_band]
        denom = a + b
        ok = valid & np.isfinite(a) & np.isfinite(b) & (denom != 0)
        if not ok.any(): continue
        idx = (a[ok] - b[ok]) / denom[ok]
        out[f"{name}_mean"] = float(idx.mean())
        out[f"{name}_median"] = float(np.median(idx))
        out[f"{name}_std"] = float(idx.std())
        out[f"{name}_p10"] = float(np.percentile(idx, 10))
        out[f"{name}_p90"] = float(np.percentile(idx, 90))
        out[f"{name}_valid_frac"] = float(ok.mean())
    total = scl.size
    for cls in SCL_REPORT:
        out[f"scl_{cls}_frac"] = float((scl == cls).sum() / total)
    return out


def collect_hansen(db, valid_ids: set[str]) -> pd.DataFrame:
    samples = {s.sample_id: s for s in db.scalars(select(Sample)) if s.sample_id in valid_ids}
    latest: dict[str, HansenAnalysis] = {}
    for a in db.scalars(select(HansenAnalysis).order_by(HansenAnalysis.created_at)):
        if a.sample_id in valid_ids: latest[a.sample_id] = a

    rows = []
    for sid, a in latest.items():
        s = samples.get(sid)
        if s is None: continue
        hist = json.loads(a.histogram_json or "[]")
        plot_area_ha = (a.plot_size_m ** 2) / 10_000.0 if a.plot_size_m else 0.0
        loss_fraction = (a.total_loss_area_ha or 0.0) / plot_area_ha if plot_area_ha > 0 else 0.0
        max_year = max((h["year"] for h in hist), default=0)
        recent_ha = sum(h["loss_area_ha"] for h in hist if h["year"] > max_year - 5)
        recent_fraction = recent_ha / plot_area_ha if plot_area_ha > 0 else 0.0
        event_year = getattr(a, "event_year", None) or getattr(a, "even_year", None)

        rows.append({
            "sample_id": s.sample_id, "lat": s.lat, "lon": s.lon,
            "driver_primary": s.driver_primary, "region": s.region,
            "source_file": s.source_file, "hansen_event_year": event_year,
            "hansen_total_loss_ha": a.total_loss_area_ha,
            "hansen_dominant_share": a.dominant_year_share,
            "hansen_loss_fraction": round(loss_fraction, 4),
            "hansen_recent_loss_5y": round(recent_fraction, 4),
            "hansen_status": a.status,
        })
    return pd.DataFrame(rows)


def collect_spectral(db, valid_ids: set[str]) -> pd.DataFrame:
    buckets = defaultdict(list)
    for d in db.scalars(select(SentinelDownload)):
        if d.sample_id not in valid_ids: continue
        period = (d.period or "").upper()
        if period not in PERIODS: continue
        tif_path = resolve_storage_path(d.local_path)
        stats = stats_for_tif(tif_path)
        if stats: buckets[(d.sample_id, period)].append(stats)

    all_keys = set()
    for items in buckets.values():
        for it in items: all_keys.update(it.keys())

    rows = defaultdict(dict)
    for (sid, period), items in buckets.items():
        for k in all_keys:
            vals = [it[k] for it in items if k in it]
            if vals: rows[sid][f"{k}_{period.lower()}"] = float(np.median(vals))

    df = pd.DataFrame.from_dict(rows, orient="index").reset_index(names="sample_id")

    delta = {}
    for col in list(df.columns):
        if not col.endswith("_pre"): continue
        base = col[:-4]
        post = f"{base}_post"
        if post in df.columns:
            delta[f"{base}_delta"] = df[post] - df[col]
            delta[f"{base}_rel"] = (df[post] - df[col]) / (df[col].abs() + 1e-6)
    if delta: df = pd.concat([df, pd.DataFrame(delta, index=df.index)], axis=1)
    return df


def collect_hansen_shape(db, valid_ids: set[str]) -> pd.DataFrame:
    latest: dict[str, HansenAnalysis] = {}
    for a in db.scalars(select(HansenAnalysis).order_by(HansenAnalysis.created_at)):
        if a.sample_id in valid_ids: latest[a.sample_id] = a

    rows = []
    for sid, a in latest.items():
        if not a.event_year: continue
        mask_row = db.scalar(
            select(HansenMask).where(HansenMask.sample_id == sid).where(HansenMask.mask_type == "dominant_year"))
        if mask_row is None: continue
        path = resolve_storage_path(mask_row.local_path)
        feats = loss_shape_features(path, event_year=a.event_year, pixel_size_m=30.0)
        if feats:
            feats["sample_id"] = sid
            rows.append(feats)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def collect_change_features(valid_ids: set[str]) -> pd.DataFrame:
    change_dir = FOREST_CHANGE_CACHE_DIR / "dnbr"
    rows = []
    for sid in valid_ids:
        json_path = change_dir / sid / "changes_localized.json"
        if not json_path.exists():
            rows.append({"sample_id": sid, "chg_matched_area_ha": 0.0, "chg_max_matched_area_ha": 0.0,
                         "chg_matched_dnbr_mean": np.nan, "chg_n_matched": 0, "chg_matched_overlap_mean": np.nan,
                         "chg_matched_compactness_mean": np.nan, "chg_max_matched_compactness": np.nan,
                         "chg_matched_rectangularity_mean": np.nan, "chg_max_matched_rectangularity": np.nan,
                         "chg_matched_solidity_mean": np.nan})
            continue
        data = json.loads(json_path.read_text(encoding="utf-8"))
        matched = [c for c in data.get("components", []) if c["matched"]]

        if matched:
            comp_vals = [c.get("compactness", 0) for c in matched]
            rect_vals = [c.get("rectangularity", 0) for c in matched]
            sol_vals = [c.get("solidity", 0) for c in matched]

            # Находим индексы самого крупного пятна для max_ признаков
            max_area_idx = np.argmax([c["area_ha"] for c in matched])

            rows.append({
                "sample_id": sid, "chg_matched_area_ha": data.get("matched_area_ha", 0.0),
                "chg_max_matched_area_ha": matched[max_area_idx]["area_ha"],
                "chg_matched_dnbr_mean": float(np.mean([c["median_dnbr"] for c in matched])),
                "chg_n_matched": len(matched),
                "chg_matched_overlap_mean": float(np.mean([c["hansen_overlap_frac"] for c in matched])),
                "chg_matched_compactness_mean": float(np.mean(comp_vals)),
                "chg_max_matched_compactness": comp_vals[max_area_idx],
                "chg_matched_rectangularity_mean": float(np.mean(rect_vals)),
                "chg_max_matched_rectangularity": rect_vals[max_area_idx],
                "chg_matched_solidity_mean": float(np.mean(sol_vals)),
            })
        else:
            rows.append({"sample_id": sid, "chg_matched_area_ha": 0.0, "chg_max_matched_area_ha": 0.0,
                         "chg_matched_dnbr_mean": np.nan, "chg_n_matched": 0, "chg_matched_overlap_mean": np.nan,
                         "chg_matched_compactness_mean": np.nan, "chg_max_matched_compactness": np.nan,
                         "chg_matched_rectangularity_mean": np.nan, "chg_max_matched_rectangularity": np.nan,
                         "chg_matched_solidity_mean": np.nan})
    return pd.DataFrame(rows)


def main() -> None:
    init_forest_database()
    FOREST_FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    with SessionLocal() as db:
        valid_ids = load_valid_ids(db)
        print(f"Valid samples: {len(valid_ids)}")
        if not valid_ids: return

        hansen = collect_hansen(db, valid_ids)
        spectral = collect_spectral(db, valid_ids)
        shape = collect_hansen_shape(db, valid_ids)
        change = collect_change_features(valid_ids)

        print(f"Hansen rows:   {len(hansen)}")
        print(f"Spectral rows: {len(spectral)}")
        print(f"Shape rows:    {len(shape)}")
        print(f"Change rows:   {len(change)}")

    df = hansen.merge(spectral, on="sample_id", how="left")
    if not shape.empty: df = df.merge(shape, on="sample_id", how="left")
    df = df.merge(change, on="sample_id", how="left")

    print(f"Merged: {len(df)} rows, {len(df.columns)} cols")
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"\n[ok] {OUT_PARQUET}")


if __name__ == "__main__":
    main()
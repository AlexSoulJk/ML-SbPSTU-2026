from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

_here = Path(__file__).resolve()
for _p in _here.parents:
    if (_p / "backend").exists():
        sys.path.insert(0, str(_p))
        break

import numpy as np
import rasterio
from sqlalchemy import select, text

from backend.config import FOREST_CHANGE_CACHE_DIR
from backend.forest.database import SessionLocal
from backend.forest.models import HansenAnalysis, SentinelDownload, SentinelSceneReview, SentinelSceneReviewReason
from backend.forest.storage_paths import resolve_storage_path

# === Конфиг ===
VALID_LABEL = "Valid"
CHANGE_MAP_DIR = FOREST_CHANGE_CACHE_DIR / "dnbr"

BAND_ORDER = ["B02", "B03", "B04", "B08", "B11", "B12", "SCL", "dataMask"]
SCL_BAD = {0, 1, 3, 8, 9, 10, 11}

# Окна вокруг события (в месяцах, отсчёт от 1 июля года события)
PRE_FROM_MONTHS = -24
PRE_TO_MONTHS = -3
POST_FROM_MONTHS = 3
POST_TO_MONTHS = 24

SEASON_MONTHS = (6, 7, 8)
MAX_SCENES_PER_PERIOD = 5
MIN_SCENES_PER_PERIOD = 1  # Если 1, то best_scene сработает
MIN_VALID_FRACTION = 0.3
STRICT_EXCLUDE_REASONS = {"CLOUDS", "NO_DATA_OR_BLACK_PIXELS"}

LIMIT_SAMPLES: int | None = None
FORCE_RECOMPUTE = False

# ВАЖНО: Выбор метода агрегации
# "median" — медиана (сглаживает шум, но может убить сигнал)
# "max_min" — max для PRE, min для POST (максимальный контраст)
# "percentile" — 75-й перцентиль для PRE, 25-й для POST
# "best_scene" — только одна ближайшая сцена
AGGREGATION_MODE = "percentile"


def load_valid_ids(db) -> set[str]:
    rows = db.execute(text("SELECT DISTINCT sample_id FROM forest_manual_validations WHERE validation = :v"),
                      {"v": VALID_LABEL}).all()
    return {r[0] for r in rows}


def read_bands(tif_path: Path) -> tuple[dict[str, np.ndarray], dict]:
    with rasterio.open(tif_path) as src:
        if src.count != len(BAND_ORDER):
            raise ValueError(f"Ожидалось {len(BAND_ORDER)} бэндов, получено {src.count}")
        bands = {name: src.read(i + 1).astype("float32") for i, name in enumerate(BAND_ORDER)}
        meta = {"transform": src.transform, "crs": src.crs, "width": src.width, "height": src.height}
    return bands, meta


def nbr_and_mask(bands: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    b08, b12 = bands["B08"], bands["B12"]
    denom = b08 + b12
    valid_d = denom != 0
    nbr = np.where(valid_d, (b08 - b12) / np.where(valid_d, denom, 1), np.nan)
    scl = bands["SCL"].astype("int32")
    dm = bands["dataMask"]
    mask = (dm > 0) & ~np.isin(scl, list(SCL_BAD)) & valid_d
    return nbr, mask


def scene_datetime(download: SentinelDownload) -> datetime | None:
    try:
        md = json.loads(download.metadata_json or "{}")
        dt = (md.get("item") or {}).get("datetime")
        if not dt: return None
        return datetime.fromisoformat(dt.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


def is_strictly_excluded(db, download_id: str) -> bool:
    reasons = set(db.scalars(
        select(SentinelSceneReviewReason.reason_code).where(SentinelSceneReviewReason.download_id == download_id)))
    return bool(reasons & STRICT_EXCLUDE_REASONS)


def aggregate_period(db, sample_id: str, period: str, event_year: int):
    event_date = datetime(event_year, 7, 1)

    if period == "PRE":
        from_date = event_date + timedelta(days=int(PRE_FROM_MONTHS * 30.4))
        to_date = event_date + timedelta(days=int(PRE_TO_MONTHS * 30.4))
    else:
        from_date = event_date + timedelta(days=int(POST_FROM_MONTHS * 30.4))
        to_date = event_date + timedelta(days=int(POST_TO_MONTHS * 30.4))

    downloads = list(db.scalars(
        select(SentinelDownload).where(SentinelDownload.sample_id == sample_id, SentinelDownload.period == period)))
    if not downloads: return None, None, None, []

    candidates = []
    for d in downloads:
        dt = scene_datetime(d)
        if not dt or not (from_date <= dt <= to_date): continue
        if SEASON_MONTHS and dt.month not in SEASON_MONTHS: continue
        if is_strictly_excluded(db, d.download_id): continue
        candidates.append((abs((dt - event_date).days), dt, d))

    if not candidates: return None, None, None, []

    # Сортируем по близости к событию
    candidates.sort(key=lambda x: x[0])
    selected = candidates[:MAX_SCENES_PER_PERIOD]

    if len(selected) < MIN_SCENES_PER_PERIOD: return None, None, None, []

    nbr_stack, mask_stack, meta, used = [], [], None, []

    for dist, dt, d in selected:
        tif_path = resolve_storage_path(d.local_path)
        if not tif_path.exists(): continue
        try:
            bands, m = read_bands(tif_path)
            nbr, mask = nbr_and_mask(bands)
            if mask.mean() < MIN_VALID_FRACTION: continue
            nbr_stack.append(nbr)
            mask_stack.append(mask)
            meta = m
            used.append({"download_id": d.download_id, "datetime": dt.strftime("%Y-%m-%d")})
        except Exception as exc:
            print(f"    [!] {d.download_id}: {exc}")

    if len(nbr_stack) < MIN_SCENES_PER_PERIOD: return None, None, None, used

    nbr_arr = np.stack(nbr_stack)
    mask_arr = np.stack(mask_stack)
    nbr_arr_masked = np.where(mask_arr, nbr_arr, np.nan)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        with np.errstate(all="ignore"):
            if AGGREGATION_MODE == "median":
                nbr_agg = np.nanmedian(nbr_arr_masked, axis=0)
            elif AGGREGATION_MODE == "max_min":
                nbr_agg = np.nanmax(nbr_arr_masked, axis=0) if period == "PRE" else np.nanmin(nbr_arr_masked, axis=0)
            elif AGGREGATION_MODE == "percentile":
                nbr_agg = np.nanpercentile(nbr_arr_masked, 75, axis=0) if period == "PRE" else np.nanpercentile(
                    nbr_arr_masked, 25, axis=0)
            elif AGGREGATION_MODE == "best_scene":
                nbr_agg = nbr_arr_masked[0]
            else:
                nbr_agg = np.nanmedian(nbr_arr_masked, axis=0)

    valid_any = mask_arr.any(axis=0)
    return nbr_agg, valid_any, meta, used


def main():
    CHANGE_MAP_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Каталог: {CHANGE_MAP_DIR}")
    print(f"AGGREGATION_MODE: {AGGREGATION_MODE}")

    with SessionLocal() as db:
        valid_ids = load_valid_ids(db)
        sample_ids = sorted(valid_ids)
        if LIMIT_SAMPLES is not None:
            sample_ids = sample_ids[:LIMIT_SAMPLES]
            print(f"Ограничение: обрабатываю первые {len(sample_ids)} samples")

        for i, sample_id in enumerate(sample_ids, 1):
            sample_dir = CHANGE_MAP_DIR / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            out_path, meta_path = sample_dir / "dnbr.tif", sample_dir / "dnbr.json"

            if not FORCE_RECOMPUTE and out_path.exists() and meta_path.exists(): continue

            hansen = db.scalar(select(HansenAnalysis).where(HansenAnalysis.sample_id == sample_id).order_by(
                HansenAnalysis.created_at.desc()))
            if not hansen or not hansen.event_year:
                print(f"  [skip] {sample_id[:12]}: нет event_year")
                continue

            try:
                nbr_pre, mask_pre, meta, used_pre = aggregate_period(db, sample_id, "PRE", hansen.event_year)
                nbr_post, mask_post, meta_post, used_post = aggregate_period(db, sample_id, "POST", hansen.event_year)

                if nbr_pre is None or nbr_post is None:
                    print(f"  [skip] {sample_id[:12]}: PRE={len(used_pre)}, POST={len(used_post)}")
                    continue

                if meta_post: meta = meta_post

                if nbr_pre.shape != nbr_post.shape:
                    h, w = min(nbr_pre.shape[0], nbr_post.shape[0]), min(nbr_pre.shape[1], nbr_post.shape[1])
                    nbr_pre, nbr_post = nbr_pre[:h, :w], nbr_post[:h, :w]
                    mask_pre, mask_post = mask_pre[:h, :w], mask_post[:h, :w]

                both_valid = mask_pre & mask_post
                dnbr = np.where(both_valid, nbr_post - nbr_pre, np.nan)

                profile = {"driver": "GTiff", "dtype": "float32", "count": 1, "width": dnbr.shape[1],
                           "height": dnbr.shape[0], "transform": meta["transform"], "crs": meta["crs"],
                           "nodata": np.nan}
                with rasterio.open(out_path, "w", **profile) as dst:
                    dst.write(dnbr.astype("float32"), 1)

                stats = {
                    "sample_id": sample_id, "event_year": int(hansen.event_year),
                    "dominant_year_share": float(hansen.dominant_year_share or 0),
                    "hansen_status": hansen.status, "n_pre_scenes": len(used_pre), "n_post_scenes": len(used_post),
                    "valid_frac": float(both_valid.mean()), "dnbr_median": float(np.nanmedian(dnbr)),
                    "dnbr_p10": float(np.nanpercentile(dnbr, 10)), "dnbr_p90": float(np.nanpercentile(dnbr, 90)),
                }
                meta_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
                (sample_dir / "scenes.json").write_text(json.dumps({"PRE": used_pre, "POST": used_post}, indent=2),
                                                        encoding="utf-8")

                print(
                    f"  [ok] {sample_id[:12]}: PRE={len(used_pre)} POST={len(used_post)} median={stats['dnbr_median']:+.3f}")

            except Exception as exc:
                print(f"  [fail] {sample_id[:12]}: {exc}")


if __name__ == "__main__":
    main()
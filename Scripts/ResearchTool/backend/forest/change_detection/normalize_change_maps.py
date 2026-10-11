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
import rasterio
from rasterio.warp import Resampling, reproject
from sqlalchemy import select

from backend.config import FOREST_CHANGE_CACHE_DIR
from backend.forest.database import SessionLocal
from backend.forest.models import HansenMask
from backend.forest.storage_paths import resolve_storage_path


CHANGE_MAP_DIR = FOREST_CHANGE_CACHE_DIR / "dnbr"

# Порог, ниже которого пиксель считается «изменением» после нормализации
CHANGE_THRESHOLD = -0.15

# Обновлять существующие файлы
FORCE_RECOMPUTE = False


def resample_hansen(hansen, hansen_meta, sentinel_meta, target_shape):
    out = np.zeros(target_shape, dtype="uint8")
    reproject(
        source=hansen.astype("uint8"),
        destination=out,
        src_transform=hansen_meta["transform"],
        src_crs=hansen_meta["crs"],
        dst_transform=sentinel_meta["transform"],
        dst_crs=sentinel_meta["crs"],
        resampling=Resampling.nearest,
    )
    return out


def process_sample(sample_dir: Path, db) -> dict | None:
    raw_path = sample_dir / "dnbr.tif"
    stats_path = sample_dir / "dnbr.json"
    norm_path = sample_dir / "dnbr_norm.tif"
    norm_stats_path = sample_dir / "dnbr_norm.json"

    if not (raw_path.exists() and stats_path.exists()):
        return None

    if norm_path.exists() and norm_stats_path.exists() and not FORCE_RECOMPUTE:
        s = json.loads(norm_stats_path.read_text(encoding="utf-8"))
        return s

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    sample_id = stats["sample_id"]

    with rasterio.open(raw_path) as src:
        dnbr = src.read(1)
        sentinel_meta = {"transform": src.transform, "crs": src.crs}
        profile = src.profile.copy()

    # Hansen mask
    mask_row = db.scalar(
        select(HansenMask)
        .where(HansenMask.sample_id == sample_id)
        .where(HansenMask.mask_type == "dominant_year")
    )
    if mask_row is None:
        return {"sample_id": sample_id, "status": "no_hansen_mask"}

    mask_path = resolve_storage_path(mask_row.local_path)
    if not mask_path.exists():
        return {"sample_id": sample_id, "status": "hansen_file_missing"}

    with rasterio.open(mask_path) as src:
        hansen = src.read(1)
        hansen_meta = {"transform": src.transform, "crs": src.crs}

    hansen_resampled = resample_hansen(hansen, hansen_meta, sentinel_meta, dnbr.shape)

    valid = np.isfinite(dnbr)
    mask_in = (hansen_resampled > 0) & valid
    mask_out = (hansen_resampled == 0) & valid

    # Фон — медиана вне Hansen-маски
    if mask_out.sum() < 50:
        # мало пикселей вне маски — берём всю медиану
        background = float(np.nanmedian(dnbr))
        background_source = "all_pixels"
    else:
        background = float(np.median(dnbr[mask_out]))
        background_source = "outside_hansen"

    # Нормализация
    dnbr_norm = dnbr - background

    # Сохраняем
    with rasterio.open(norm_path, "w", **profile) as dst:
        dst.write(dnbr_norm.astype("float32"), 1)

    # Сколько пикселей внутри маски ниже порога
    inside_change = int(((dnbr_norm < CHANGE_THRESHOLD) & mask_in).sum())
    inside_total = int(mask_in.sum())
    outside_change = int(((dnbr_norm < CHANGE_THRESHOLD) & mask_out).sum())
    outside_total = int(mask_out.sum())

    norm_stats = {
        "sample_id": sample_id,
        "event_year": stats.get("event_year"),
        "hansen_status": stats.get("hansen_status"),
        "dominant_share": stats.get("dominant_year_share"),
        "n_pre": stats.get("n_pre_scenes"),
        "n_post": stats.get("n_post_scenes"),
        "background": background,
        "background_source": background_source,
        "n_pixels_valid": int(valid.sum()),
        "n_inside_mask": inside_total,
        "n_outside_mask": outside_total,
        "inside_change_frac": round(inside_change / max(inside_total, 1), 3),
        "outside_change_frac": round(outside_change / max(outside_total, 1), 3),
        "dnbr_norm_min": float(np.nanmin(dnbr_norm)),
        "dnbr_norm_p10": float(np.nanpercentile(dnbr_norm, 10)),
        "dnbr_norm_median": float(np.nanmedian(dnbr_norm)),
        "dnbr_norm_p90": float(np.nanpercentile(dnbr_norm, 90)),
        "dnbr_norm_max": float(np.nanmax(dnbr_norm)),
    }
    norm_stats_path.write_text(
        json.dumps(norm_stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return norm_stats


def main() -> None:
    if not CHANGE_MAP_DIR.exists():
        print(f"Нет каталога {CHANGE_MAP_DIR}")
        return

    sample_dirs = sorted([p for p in CHANGE_MAP_DIR.iterdir() if p.is_dir()])
    print(f"Найдено samples: {len(sample_dirs)}")
    print(f"Порог изменения: dNBR_norm < {CHANGE_THRESHOLD}")
    print()

    results = []
    with SessionLocal() as db:
        for i, sample_dir in enumerate(sample_dirs, 1):
            try:
                r = process_sample(sample_dir, db)
                if r is not None:
                    results.append(r)
            except Exception as exc:
                print(f"  [fail] {sample_dir.name[:12]}: {exc}")
            if i % 20 == 0:
                print(f"  обработано: {i}/{len(sample_dirs)}")

    ok = [r for r in results if r.get("status") != "no_hansen_mask"]
    print(f"\n=== ГОТОВО: {len(ok)} samples ===")

    if ok:
        # Фон
        backgrounds = [r["background"] for r in ok]
        print(f"Фон (медиана вне маски): медиана={np.median(backgrounds):+.3f}, "
              f"диапазон [{min(backgrounds):+.3f}, {max(backgrounds):+.3f}]")

        # Доля изменений внутри/вне
        inside_fracs = [r["inside_change_frac"] for r in ok]
        outside_fracs = [r["outside_change_frac"] for r in ok]
        print(f"\nДоля пикселей с dNBR_norm < {CHANGE_THRESHOLD}:")
        print(f"  внутри маски: медиана={np.median(inside_fracs):.3f}, "
              f"диапазон [{min(inside_fracs):.3f}, {max(inside_fracs):.3f}]")
        print(f"  вне маски:    медиана={np.median(outside_fracs):.3f}, "
              f"диапазон [{min(outside_fracs):.3f}, {max(outside_fracs):.3f}]")

        ratio = np.array(inside_fracs) / np.array([max(o, 1e-3) for o in outside_fracs])
        good = (ratio > 3).sum()
        bad = (ratio < 1.5).sum()
        print(f"\nОтношение inside/outside: >3 у {good}, <1.5 у {bad}, "
              f"промежуточных {len(ok) - good - bad}")

        # Сводка по годам
        print("\n=== ПО ГОДАМ ===")
        import pandas as pd
        df = pd.DataFrame(ok)
        print(df.groupby("event_year").agg(
            n=("sample_id", "count"),
            bg=("background", "median"),
            in_frac=("inside_change_frac", "median"),
            out_frac=("outside_change_frac", "median"),
        ).round(3).to_string())

        # CSV
        from backend.config import FOREST_MODELS_DIR
        FOREST_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = FOREST_MODELS_DIR / "change_map_normalized.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\n[ok] {csv_path}")


if __name__ == "__main__":
    main()
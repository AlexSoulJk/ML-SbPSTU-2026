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
import rasterio
from rasterio.warp import Resampling, reproject
from sqlalchemy import select

from backend.config import FOREST_CHANGE_CACHE_DIR, FOREST_MODELS_DIR
from backend.forest.database import SessionLocal
from backend.forest.models import HansenMask
from backend.forest.storage_paths import resolve_storage_path


CHANGE_MAP_DIR = FOREST_CHANGE_CACHE_DIR / "dnbr"

# Какую карту использовать: "dnbr" или "dnbr_norm"
MAP_VARIANT = "dnbr_norm"

# Порог изменения dNBR (пиксель считается «изменённым», если ниже)
CHANGE_THRESHOLD = -0.15

# Пороги вердикта по отношению inside/outside
RATIO_CONSISTENT = 2.0
RATIO_BORDERLINE = 1.4

# Сколько PNG-визуализаций сохранить по каждому вердикту
N_EXAMPLES_PER_CLASS = 3

# Диапазон для z-score визуализации
ZSCORE_VMIN = -3.0
ZSCORE_VMAX = 3.0


# === Вспомогательные ===
def resample_hansen_to_sentinel(hansen, hansen_meta, sentinel_meta, target_shape):
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


def load_hansen_mask(sample_id: str, db):
    mask_row = db.scalar(
        select(HansenMask)
        .where(HansenMask.sample_id == sample_id)
        .where(HansenMask.mask_type == "dominant_year")
    )
    if mask_row is None:
        return None
    mask_path = resolve_storage_path(mask_row.local_path)
    if not mask_path.exists():
        return None
    with rasterio.open(mask_path) as src:
        hansen = src.read(1)
        meta = {"transform": src.transform, "crs": src.crs}
    return hansen, meta


def inspect_one(sample_dir: Path, db) -> dict | None:
    tif_path = sample_dir / f"{MAP_VARIANT}.tif"
    stats_path = sample_dir / "dnbr.json"     # метаданные всегда из исходного
    if not (tif_path.exists() and stats_path.exists()):
        return None

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    sample_id = stats["sample_id"]

    with rasterio.open(tif_path) as src:
        dnbr = src.read(1)
        sentinel_meta = {"transform": src.transform, "crs": src.crs}

    hansen_pair = load_hansen_mask(sample_id, db)
    if hansen_pair is None:
        return {"sample_id": sample_id, "status": "no_hansen_mask"}
    hansen, hansen_meta = hansen_pair

    hansen_resampled = resample_hansen_to_sentinel(
        hansen, hansen_meta, sentinel_meta, dnbr.shape,
    )

    valid = np.isfinite(dnbr)
    mask_in = (hansen_resampled > 0) & valid
    mask_out = (hansen_resampled == 0) & valid

    n_in = int(mask_in.sum())
    n_out = int(mask_out.sum())

    if n_in < 10 or n_out < 10:
        return {
            "sample_id": sample_id,
            "status": "too_few_pixels",
            "n_in": n_in,
            "n_out": n_out,
            "event_year": stats.get("event_year"),
        }

    d_in = dnbr[mask_in]
    d_out = dnbr[mask_out]

    inside_change_frac = float((d_in < CHANGE_THRESHOLD).mean())
    outside_change_frac = float((d_out < CHANGE_THRESHOLD).mean())
    ratio = inside_change_frac / max(outside_change_frac, 1e-3)

    if ratio >= RATIO_CONSISTENT:
        verdict = "consistent"
    elif ratio >= RATIO_BORDERLINE:
        verdict = "borderline"
    else:
        verdict = "no_signal"

    norm_stats = {}
    norm_stats_path = sample_dir / "dnbr_norm.json"
    if norm_stats_path.exists():
        try:
            norm_stats = json.loads(norm_stats_path.read_text(encoding="utf-8"))
        except Exception:
            norm_stats = {}

    return {
        "sample_id": sample_id,
        "status": "ok",
        "event_year": stats.get("event_year"),
        "hansen_status": stats.get("hansen_status"),
        "dominant_share": stats.get("dominant_year_share"),
        "n_pre": stats.get("n_pre_scenes"),
        "n_post": stats.get("n_post_scenes"),
        "valid_frac": stats.get("valid_frac"),
        "background": norm_stats.get("background"),
        "n_in": n_in,
        "n_out": n_out,
        "med_in": float(np.median(d_in)),
        "med_out": float(np.median(d_out)),
        "inside_change_frac": round(inside_change_frac, 3),
        "outside_change_frac": round(outside_change_frac, 3),
        "ratio": round(ratio, 3),
        "verdict": verdict,
    }


def save_png(sample_dir: Path, db, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    stats = json.loads((sample_dir / "dnbr.json").read_text(encoding="utf-8"))
    sample_id = stats["sample_id"]

    with rasterio.open(sample_dir / f"{MAP_VARIANT}.tif") as src:
        dnbr = src.read(1)
        sentinel_meta = {"transform": src.transform, "crs": src.crs}

    hansen_pair = load_hansen_mask(sample_id, db)
    if hansen_pair is None:
        return
    hansen, hansen_meta = hansen_pair

    hansen_resampled = resample_hansen_to_sentinel(
        hansen, hansen_meta, sentinel_meta, dnbr.shape,
    )

    # z-score по AOI
    finite = np.isfinite(dnbr)
    if finite.sum() > 0:
        mean = float(np.nanmean(dnbr))
        std = float(np.nanstd(dnbr)) or 1e-6
        dnbr_z = (dnbr - mean) / std
    else:
        mean, std = 0.0, 1.0
        dnbr_z = dnbr

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(dnbr, cmap="RdBu_r", vmin=-0.6, vmax=0.6)
    axes[0].set_title(f"{MAP_VARIANT}\nmedian={np.nanmedian(dnbr):+.3f}")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(dnbr_z, cmap="RdBu_r", vmin=ZSCORE_VMIN, vmax=ZSCORE_VMAX)
    axes[1].set_title(f"z-score (mean={mean:+.3f}, std={std:.3f})")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    axes[2].imshow(dnbr_z, cmap="RdBu_r", vmin=ZSCORE_VMIN, vmax=ZSCORE_VMAX)
    axes[2].contour(hansen_resampled > 0, colors="black", linewidths=0.5)
    axes[2].set_title("z-score + контур Hansen")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()


def main() -> None:
    if not CHANGE_MAP_DIR.exists():
        print(f"Нет каталога {CHANGE_MAP_DIR}")
        return

    print(f"MAP_VARIANT = {MAP_VARIANT}")
    print(f"CHANGE_THRESHOLD = {CHANGE_THRESHOLD}")
    print(f"RATIO_CONSISTENT >= {RATIO_CONSISTENT}, RATIO_BORDERLINE >= {RATIO_BORDERLINE}")

    sample_dirs = sorted([p for p in CHANGE_MAP_DIR.iterdir() if p.is_dir()])
    print(f"Найдено samples: {len(sample_dirs)}\n")

    rows = []
    with SessionLocal() as db:
        for i, sample_dir in enumerate(sample_dirs, 1):
            try:
                result = inspect_one(sample_dir, db)
                if result is not None:
                    rows.append(result)
            except Exception as exc:
                rows.append({
                    "sample_id": sample_dir.name,
                    "status": f"error: {exc}",
                })
            if i % 20 == 0:
                print(f"  обработано: {i}/{len(sample_dirs)}")

    df = pd.DataFrame(rows)

    print("\n=== СВОДКА ПО СТАТУСАМ ===")
    print(df["status"].value_counts().to_string())

    ok = df[df["status"] == "ok"]
    if not ok.empty:
        print("\n=== ВЕРДИКТЫ ===")
        print(ok["verdict"].value_counts().to_string())

        print("\n=== ДОЛИ ИЗМЕНЕНИЯ ===")
        print(f"inside_change_frac:  медиана={ok['inside_change_frac'].median():.3f}, "
              f"диапазон [{ok['inside_change_frac'].min():.3f}, {ok['inside_change_frac'].max():.3f}]")
        print(f"outside_change_frac: медиана={ok['outside_change_frac'].median():.3f}, "
              f"диапазон [{ok['outside_change_frac'].min():.3f}, {ok['outside_change_frac'].max():.3f}]")
        print(f"ratio:               медиана={ok['ratio'].median():.3f}, "
              f"диапазон [{ok['ratio'].min():.3f}, {ok['ratio'].max():.3f}]")

        if "background" in ok.columns:
            bg = ok["background"].dropna()
            if len(bg) > 0:
                print(f"background:          медиана={bg.median():+.3f}, "
                      f"диапазон [{bg.min():+.3f}, {bg.max():+.3f}]")

        print("\n=== ПО event_year ===")
        print(ok.groupby("event_year").agg(
            n=("sample_id", "count"),
            n_consistent=("verdict", lambda s: (s == "consistent").sum()),
            n_borderline=("verdict", lambda s: (s == "borderline").sum()),
            n_no_signal=("verdict", lambda s: (s == "no_signal").sum()),
            med_ratio=("ratio", "median"),
        ).to_string())

        print("\n=== ПО hansen_status ===")
        print(ok.groupby("hansen_status").agg(
            n=("sample_id", "count"),
            n_consistent=("verdict", lambda s: (s == "consistent").sum()),
            med_ratio=("ratio", "median"),
        ).to_string())

        # PNG для примеров
        for verdict in ("consistent", "borderline", "no_signal"):
            subset = ok[ok["verdict"] == verdict]
            if subset.empty:
                continue
            print(f"\n=== Примеры: {verdict} ({len(subset)} samples) ===")
            cols = ["sample_id", "event_year", "hansen_status",
                    "inside_change_frac", "outside_change_frac", "ratio"]
            print(subset[cols].head(5).to_string(index=False))

            for _, row in subset.head(N_EXAMPLES_PER_CLASS).iterrows():
                sample_dir = CHANGE_MAP_DIR / row["sample_id"]
                out_png = sample_dir / f"inspect_{MAP_VARIANT}_{verdict}.png"
                if out_png.exists():
                    continue
                try:
                    with SessionLocal() as db:
                        save_png(sample_dir, db, out_png)
                    print(f"  PNG: {out_png.name}")
                except Exception as exc:
                    print(f"  PNG не сохранён для {row['sample_id']}: {exc}")

    FOREST_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = FOREST_MODELS_DIR / f"change_map_diagnostics_{MAP_VARIANT}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n[ok] {csv_path}")


if __name__ == "__main__":
    main()
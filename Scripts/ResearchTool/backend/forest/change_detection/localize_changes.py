from __future__ import annotations

import json
import sys
from pathlib import Path

_here = Path(__file__).resolve()
for _p in _here.parents:
    if (_p / "backend").exists():
        sys.path.insert(0, str(_p))
        break

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject
from scipy.ndimage import gaussian_filter, binary_dilation
from skimage.measure import label, regionprops
from sqlalchemy import select

from backend.config import FOREST_CHANGE_CACHE_DIR, FOREST_MODELS_DIR
from backend.forest.database import SessionLocal
from backend.forest.models import HansenMask
from backend.forest.storage_paths import resolve_storage_path

CHANGE_MAP_DIR = FOREST_CHANGE_CACHE_DIR / "dnbr"
MAP_VARIANT = "dnbr_norm"
CHANGE_THRESHOLD = -0.10
MIN_COMPONENT_PX = 20
HANSEN_OVERLAP_GOOD = 0.3
LIMIT_SAMPLES: int | None = None  # None = все 118


def resample_hansen(hansen, hansen_meta, sentinel_meta, target_shape):
    out = np.zeros(target_shape, dtype="uint8")
    reproject(source=hansen.astype("uint8"), destination=out, src_transform=hansen_meta["transform"],
              src_crs=hansen_meta["crs"],
              dst_transform=sentinel_meta["transform"], dst_crs=sentinel_meta["crs"], resampling=Resampling.nearest)
    return out


def load_hansen_mask(sample_id, db):
    row = db.scalar(
        select(HansenMask).where(HansenMask.sample_id == sample_id).where(HansenMask.mask_type == "dominant_year"))
    if not row: return None
    path = resolve_storage_path(row.local_path)
    if not path.exists(): return None
    with rasterio.open(path) as src:
        return src.read(1), {"transform": src.transform, "crs": src.crs}


def plot_and_save(sample_dir: Path, sample_id: str, dnbr: np.ndarray, hansen_resampled: np.ndarray,
                  change_mask: np.ndarray, labeled: np.ndarray, props, n_matched: int, n_unmatched: int):
    rgb = np.ones((*dnbr.shape, 3))
    rgb[~change_mask] = [0.95, 0.95, 0.95]
    rgb[change_mask] = [0.8, 0.2, 0.2]

    for p in props:
        if p.area < MIN_COMPONENT_PX: continue
        comp_mask = (labeled == p.label)
        if (comp_mask & (hansen_resampled > 0)).sum() / p.area >= HANSEN_OVERLAP_GOOD:
            rgb[comp_mask] = [0.1, 0.8, 0.1]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    im0 = axes[0].imshow(dnbr, cmap="RdBu_r", vmin=-0.6, vmax=0.6)
    axes[0].contour(hansen_resampled > 0, colors="black", linewidths=1.5)
    axes[0].set_title(f"{sample_id[:12]}\ndNBR + Hansen")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    axes[1].imshow(change_mask, cmap="gray", alpha=0.8)
    axes[1].contour(hansen_resampled > 0, colors="red", linewidths=1.5)
    axes[1].set_title(f"Изменения (dNBR < {CHANGE_THRESHOLD})")

    axes[2].imshow(rgb)
    axes[2].contour(hansen_resampled > 0, colors="black", linewidths=1.5)
    axes[2].set_title(f"Компоненты (matched={n_matched}, unmatched={n_unmatched})")

    for ax in axes: ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(sample_dir / "localized_changes.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def analyze_sample(sample_dir: Path, db) -> dict | None:
    tif_path, stats_path = sample_dir / f"{MAP_VARIANT}.tif", sample_dir / "dnbr.json"
    if not (tif_path.exists() and stats_path.exists()): return None

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    sample_id = stats["sample_id"]

    with rasterio.open(tif_path) as src:
        dnbr = src.read(1)
        sentinel_meta = {"transform": src.transform, "crs": src.crs}

    hansen_pair = load_hansen_mask(sample_id, db)
    if not hansen_pair: return None
    hansen, hansen_meta = hansen_pair

    hansen_resampled = resample_hansen(hansen, hansen_meta, sentinel_meta, dnbr.shape)
    hansen_buffered = binary_dilation(hansen_resampled > 0, iterations=2)

    dnbr_smooth = gaussian_filter(dnbr, sigma=1.5)
    change_mask = (dnbr_smooth < CHANGE_THRESHOLD) & np.isfinite(dnbr_smooth)

    labeled = label(change_mask, connectivity=2)
    props = regionprops(labeled)

    components = []
    n_matched, n_unmatched = 0, 0

    for p in props:
        if p.area < MIN_COMPONENT_PX: continue
        comp_mask = (labeled == p.label)

        overlap_in = (comp_mask & hansen_buffered).sum()
        overlap_frac = overlap_in / p.area

        is_matched = overlap_frac >= HANSEN_OVERLAP_GOOD
        if is_matched:
            n_matched += 1
        else:
            n_unmatched += 1

        # Расчёт формы
        compactness = float(4 * np.pi * p.area / max(p.perimeter, 1e-6) ** 2)
        bbox_h = max(p.bbox[2] - p.bbox[0], 1)
        bbox_w = max(p.bbox[3] - p.bbox[1], 1)
        rectangularity = float(p.area / (bbox_h * bbox_w))
        solidity = float(p.solidity)

        components.append({
            "area_px": int(p.area),
            "area_ha": float(p.area * 100 / 10_000),
            "median_dnbr": float(np.median(dnbr[comp_mask])),
            "hansen_overlap_frac": round(float(overlap_frac), 3),
            "matched": bool(is_matched),
            "compactness": round(compactness, 3),
            "rectangularity": round(rectangularity, 3),
            "solidity": round(solidity, 3),
        })

    try:
        plot_and_save(sample_dir, sample_id, dnbr, hansen_resampled, change_mask, labeled, props, n_matched,
                      n_unmatched)
    except Exception as exc:
        print(f"  [!] PNG fail {sample_id[:12]}: {exc}")

    matched = [c for c in components if c["matched"]]
    unmatched = [c for c in components if not c["matched"]]

    return {
        "sample_id": sample_id, "event_year": stats.get("event_year"),
        "n_components": len(components), "n_matched": len(matched), "n_unmatched": len(unmatched),
        "total_changed_area_ha": round(sum(c["area_ha"] for c in components), 3),
        "matched_area_ha": round(sum(c["area_ha"] for c in matched), 3),
        "unmatched_area_ha": round(sum(c["area_ha"] for c in unmatched), 3),
        "components": components,
    }


def main():
    sample_dirs = sorted([p for p in CHANGE_MAP_DIR.iterdir() if p.is_dir() and not p.name.startswith("_")])
    if LIMIT_SAMPLES is not None: sample_dirs = sample_dirs[:LIMIT_SAMPLES]
    print(f"Обрабатываю {len(sample_dirs)} samples\n")

    results = []
    with SessionLocal() as db:
        for i, d in enumerate(sample_dirs, 1):
            try:
                r = analyze_sample(d, db)
                if r is not None:
                    results.append(r)
                    (d / "changes_localized.json").write_text(json.dumps(r, indent=2, ensure_ascii=False),
                                                              encoding="utf-8")
            except Exception as exc:
                print(f"  [fail] {d.name[:12]}: {exc}")
            if i % 20 == 0: print(f"  обработано {i}/{len(sample_dirs)}")

    print("\n=== СВОДКА ===")
    if results:
        print(f"Сэмплов с matched-компонентами: {sum(1 for r in results if r['n_matched'] > 0)}/{len(results)}")
        print(f"Всего площадь matched:   {sum(r['matched_area_ha'] for r in results):.2f} га")
        print(f"Всего площадь unmatched: {sum(r['unmatched_area_ha'] for r in results):.2f} га")

    FOREST_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    flat = [{"sample_id": r["sample_id"], "event_year": r["event_year"], **c} for r in results for c in r["components"]]
    pd.DataFrame(flat).to_csv(FOREST_MODELS_DIR / "localized_changes.csv", index=False, encoding="utf-8-sig")
    print(f"\n[ok] {FOREST_MODELS_DIR / 'localized_changes.csv'}")


if __name__ == "__main__":
    main()
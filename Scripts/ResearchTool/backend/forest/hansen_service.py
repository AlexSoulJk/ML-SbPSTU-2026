from __future__ import annotations

import json
import math
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np
import rasterio
from PIL import Image
from rasterio.mask import mask as rio_mask
from rasterio.transform import array_bounds
from rasterio.warp import transform_bounds, transform_geom
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import FOREST_CACHE_DIR, FOREST_HANSEN_CACHE_DIR
from .geometry import DEFAULT_SAMPLE_PLOT_M, sample_geometries
from .hansen_provider import HANSEN_VERSION, download_hansen_tile, tile_id_for_lonlat, tile_model
from .models import HansenAnalysis, HansenMask, HansenTile, Sample, SampleStatus, utc_now


ProgressCallback = Callable[[str, str], None]
CancelCallback = Callable[[], bool]
TOO_OLD_EVENT_YEAR_CUTOFF = 2016

def existing_hansen_analysis(db: Session, sample_id: str) -> HansenAnalysis | None:
    return db.scalar(
        select(HansenAnalysis)
        .where(HansenAnalysis.sample_id == sample_id)
        .order_by(HansenAnalysis.created_at.desc())
        .limit(1)
    )

def report(progress: ProgressCallback | None, stage: str, message: str) -> None:
    if progress:
        progress(stage, message)


def raise_if_cancelled(should_cancel: CancelCallback | None) -> None:
    if should_cancel and should_cancel():
        raise RuntimeError("Hansen analysis cancelled.")


def pixel_area_ha(transform: Any, crs: Any, lat: float) -> float:
    width = abs(float(transform.a))
    height = abs(float(transform.e))
    if crs and getattr(crs, "is_geographic", False):
        width_m = width * 111_320.0 * math.cos(math.radians(lat))
        height_m = height * 110_574.0
    else:
        width_m = width
        height_m = height
    return max(0.0, width_m * height_m / 10_000.0)


def lonlat_bbox(transform: Any, width: int, height: int, crs: Any) -> list[float]:
    left, bottom, right, top = array_bounds(height, width, transform)
    if crs and str(crs).upper() not in {"EPSG:4326", "OGC:CRS84"}:
        left, bottom, right, top = transform_bounds(crs, "EPSG:4326", left, bottom, right, top)
    return [left, bottom, right, top]


def rgba_loss_year(loss: np.ndarray) -> np.ndarray:
    data = np.asarray(loss, dtype=np.uint8)
    alpha = np.where(data > 0, 190, 0).astype(np.uint8)
    norm = np.clip((data.astype(np.float32) - 1.0) / 25.0, 0.0, 1.0)
    rgba = np.zeros((*data.shape, 4), dtype=np.uint8)
    rgba[..., 0] = np.where(data > 0, 244, 0)
    rgba[..., 1] = np.where(data > 0, (190 - norm * 135).astype(np.uint8), 0)
    rgba[..., 2] = np.where(data > 0, 58, 0)
    rgba[..., 3] = alpha
    return rgba


def rgba_binary(mask: np.ndarray, color: tuple[int, int, int], alpha: int = 210) -> np.ndarray:
    data = np.asarray(mask, dtype=bool)
    rgba = np.zeros((*data.shape, 4), dtype=np.uint8)
    rgba[..., 0] = color[0]
    rgba[..., 1] = color[1]
    rgba[..., 2] = color[2]
    rgba[..., 3] = np.where(data, alpha, 0).astype(np.uint8)
    return rgba


def rgba_treecover(treecover: np.ndarray) -> np.ndarray:
    data = np.clip(np.asarray(treecover, dtype=np.float32), 0, 100)
    rgba = np.zeros((*data.shape, 4), dtype=np.uint8)
    rgba[..., 0] = 35
    rgba[..., 1] = 130
    rgba[..., 2] = 82
    rgba[..., 3] = np.where(data > 0, np.clip(data * 1.8, 35, 180), 0).astype(np.uint8)
    return rgba


def write_raster(path: Path, data: np.ndarray, profile: dict[str, Any], dtype: str = "uint8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out_profile = profile.copy()
    out_profile.update(
        count=1,
        dtype=dtype,
        nodata=0,
        compress="deflate",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
    )
    with rasterio.open(path, "w", **out_profile) as dataset:
        dataset.write(data.astype(dtype), 1)


def write_png(path: Path, rgba: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(path)


def cache_url(path: Path) -> str | None:
    try:
        relative = path.relative_to(FOREST_CACHE_DIR)
    except ValueError:
        return None
    return "/cache/forest/" + "/".join(relative.parts)


def status_for_loss(total_loss_area_ha: float, dominant_area_ha: float, share: float) -> str:
    if total_loss_area_ha == 0:
        return "NO_LOSS"
    if share >= 0.60 and dominant_area_ha >= 1.0:
        return "GOOD"
    return "AMBIGUOUS"


def upsert_tile(db: Session, layer: str, tile_id: str, local_path: str) -> None:
    existing = db.scalar(
        select(HansenTile).where(
            HansenTile.layer == layer,
            HansenTile.tile_id == tile_id,
            HansenTile.version == HANSEN_VERSION,
        )
    )
    if existing is None:
        db.add(tile_model(layer, tile_id, local_path))
        return
    existing.local_path = local_path


def upsert_hansen_status(
    db: Session,
    sample_id: str,
    status: str,
    metrics: dict[str, Any],
) -> None:
    existing = db.scalar(
        select(SampleStatus).where(
            SampleStatus.sample_id == sample_id,
            SampleStatus.source == "hansen",
        )
    )
    if existing is None:
        existing = SampleStatus(sample_id=sample_id, source="hansen", status=status)
        db.add(existing)
    existing.status = status
    existing.reason = metrics.get("reason")
    existing.metrics_json = json.dumps(metrics, ensure_ascii=False, sort_keys=True)
    existing.updated_at = utc_now()


def read_masked_layer(path: str, geometry_lonlat: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    with rasterio.open(path) as dataset:
        geometry = geometry_lonlat
        if dataset.crs and str(dataset.crs).upper() not in {"EPSG:4326", "OGC:CRS84"}:
            geometry = transform_geom("EPSG:4326", dataset.crs, geometry_lonlat)
        data, transform = rio_mask(dataset, [geometry], crop=True, filled=True, nodata=0)
        profile = dataset.profile.copy()
        profile.update(transform=transform, height=data.shape[1], width=data.shape[2])
        return data[0], profile


def latest_hansen_payload(db: Session, sample_id: str) -> dict[str, Any] | None:
    analysis = db.scalar(
        select(HansenAnalysis)
        .where(HansenAnalysis.sample_id == sample_id)
        .order_by(HansenAnalysis.created_at.desc())
        .limit(1)
    )
    if analysis is None:
        return None

    masks = list(
        db.scalars(select(HansenMask).where(HansenMask.analysis_id == analysis.analysis_id))
    )
    return {
        "analysis_id": analysis.analysis_id,
        "event_year": analysis.event_year,
        "total_loss_area_ha": analysis.total_loss_area_ha,
        "dominant_loss_area_ha": analysis.dominant_loss_area_ha,
        "dominant_year_share": analysis.dominant_year_share,
        "status": analysis.status,
        "histogram": json.loads(analysis.histogram_json),
        "created_at": analysis.created_at.isoformat() if analysis.created_at else None,
        "masks": [
            {
                "mask_id": mask.mask_id,
                "mask_type": mask.mask_type,
                "local_path": mask.local_path,
                "png_url": cache_url(Path(mask.local_path).with_suffix(".png")),
                "bbox": json.loads(mask.bbox_json),
            }
            for mask in masks
        ],
    }


def analyze_sample_hansen(
    db: Session,
    sample: Sample,
    *,
    include_treecover: bool = False,
    skip_existing: bool = True,
    progress: ProgressCallback | None = None,
    should_cancel: CancelCallback | None = None,
) -> dict[str, Any]:
    if skip_existing:
        existing = existing_hansen_analysis(db, sample.sample_id)
        if existing is not None:
            report(
                progress,
                "done",
                f"Hansen already exists for {sample.sample_id[:8]}, skipping",
            )
            return latest_hansen_payload(db, sample.sample_id) or {}

    report(progress, "loading", f"Loading sample {sample.sample_id[:8]}")
    raise_if_cancelled(should_cancel)

    geometries = sample_geometries(sample.lon, sample.lat)
    sample_plot = geometries["sample_plot"]
    tile_id = tile_id_for_lonlat(sample.lon, sample.lat)

    report(progress, "cache", f"Checking Hansen cache for tile {tile_id}")
    lossyear_path = download_hansen_tile(
        "lossyear",
        tile_id,
        progress=progress,
        should_cancel=should_cancel,
    )
    raise_if_cancelled(should_cancel)
    upsert_tile(db, "lossyear", tile_id, lossyear_path)

    treecover_path = None
    if include_treecover:
        treecover_path = download_hansen_tile(
            "treecover2000",
            tile_id,
            progress=progress,
            should_cancel=should_cancel,
        )
        raise_if_cancelled(should_cancel)
        upsert_tile(db, "treecover2000", tile_id, treecover_path)

    report(progress, "analysis", "Reading Hansen raster windows")
    loss, loss_profile = read_masked_layer(lossyear_path, sample_plot)
    treecover = None
    treecover_profile = None
    if treecover_path:
        treecover, treecover_profile = read_masked_layer(treecover_path, sample_plot)
    raise_if_cancelled(should_cancel)
    viewer_aoi = geometries["viewer_aoi"]
    report(progress, "analysis", "Reading Hansen AOI context")
    aoi_loss, aoi_loss_profile = read_masked_layer(lossyear_path, viewer_aoi)
    aoi_treecover = None
    aoi_treecover_profile = None
    if treecover_path:
        aoi_treecover, aoi_treecover_profile = read_masked_layer(treecover_path, viewer_aoi)
    raise_if_cancelled(should_cancel)

    report(progress, "analysis", "Computing loss histogram")
    area_ha = pixel_area_ha(loss_profile["transform"], loss_profile.get("crs"), sample.lat)
    loss_codes, counts = np.unique(loss[loss > 0], return_counts=True)
    histogram = [
        {
            "year": 2000 + int(code),
            "pixels": int(count),
            "loss_area_ha": round(float(count) * area_ha, 4),
        }
        for code, count in zip(loss_codes.tolist(), counts.tolist(), strict=True)
    ]

    total_loss_area_ha = round(sum(item["loss_area_ha"] for item in histogram), 4)
    dominant = max(histogram, key=lambda item: item["loss_area_ha"], default=None)
    event_year = dominant["year"] if dominant else None
    dominant_area = float(dominant["loss_area_ha"]) if dominant else 0.0
    share = round(dominant_area / total_loss_area_ha, 4) if total_loss_area_ha > 0 else 0.0
    status = (
        "TOO_OLD"
        if event_year is not None and event_year < TOO_OLD_EVENT_YEAR_CUTOFF
        else status_for_loss(total_loss_area_ha, dominant_area, share)
    )

    analysis_id = uuid.uuid4().hex
    analysis = HansenAnalysis(
        analysis_id=analysis_id,
        sample_id=sample.sample_id,
        analysis_area_type="sample_plot",
        plot_size_m=DEFAULT_SAMPLE_PLOT_M,
        treecover_threshold=None,
        event_year=event_year,
        total_loss_area_ha=total_loss_area_ha,
        dominant_loss_area_ha=dominant_area,
        dominant_year_share=share,
        status=status,
        histogram_json=json.dumps(histogram, ensure_ascii=False, sort_keys=True),
    )
    db.add(analysis)

    mask_dir = FOREST_HANSEN_CACHE_DIR / "masks" / sample.sample_id / analysis_id
    report(progress, "analysis", "Writing Hansen masks")
    mask_specs = [
        ("all_loss", loss.astype(np.uint8), rgba_loss_year(loss), loss_profile),
        (
            "dominant_year",
            (loss == event_year - 2000).astype(np.uint8) if event_year else np.zeros_like(loss),
            rgba_binary(loss == event_year - 2000, (220, 70, 45)) if event_year else rgba_binary(loss, (220, 70, 45), 0),
            loss_profile,
        ),
        ("aoi_all_loss", aoi_loss.astype(np.uint8), rgba_loss_year(aoi_loss), aoi_loss_profile),
        (
            "aoi_dominant_year",
            (aoi_loss == event_year - 2000).astype(np.uint8) if event_year else np.zeros_like(aoi_loss),
            rgba_binary(aoi_loss == event_year - 2000, (220, 70, 45)) if event_year else rgba_binary(aoi_loss, (220, 70, 45), 0),
            aoi_loss_profile,
        ),
    ]
    if treecover is not None and treecover_profile is not None:
        mask_specs.append(
            ("treecover2000", treecover.astype(np.uint8), rgba_treecover(treecover), treecover_profile)
        )
    if aoi_treecover is not None and aoi_treecover_profile is not None:
        mask_specs.append(
            (
                "aoi_treecover2000",
                aoi_treecover.astype(np.uint8),
                rgba_treecover(aoi_treecover),
                aoi_treecover_profile,
            )
        )

    for mask_type, raster_data, png_data, profile in mask_specs:
        raise_if_cancelled(should_cancel)
        tif_path = mask_dir / f"{mask_type}.tif"
        png_path = tif_path.with_suffix(".png")
        layer_bbox = lonlat_bbox(
            profile["transform"],
            profile["width"],
            profile["height"],
            profile.get("crs"),
        )
        write_raster(tif_path, raster_data, profile)
        write_png(png_path, png_data)
        db.add(
            HansenMask(
                mask_id=uuid.uuid4().hex,
                sample_id=sample.sample_id,
                analysis_id=analysis_id,
                mask_type=mask_type,
                local_path=str(tif_path),
                bbox_json=json.dumps(layer_bbox),
            )
        )

    metrics = {
        "event_year": event_year,
        "total_loss_area_ha": total_loss_area_ha,
        "dominant_loss_area_ha": dominant_area,
        "dominant_year_share": share,
        "tile_id": tile_id,
        "include_treecover": include_treecover,
        "too_old_cutoff_year": TOO_OLD_EVENT_YEAR_CUTOFF,
    }
    upsert_hansen_status(db, sample.sample_id, status, metrics)
    db.commit()
    report(progress, "done", f"Hansen ready: {status}, event {event_year or 'n/a'}")
    return latest_hansen_payload(db, sample.sample_id) or {}

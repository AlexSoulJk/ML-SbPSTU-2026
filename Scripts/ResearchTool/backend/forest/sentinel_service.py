from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..cdse_client import (
    S2_RAW_EVALSCRIPT,
    process_request,
    run_process_api,
    stac_search_s2,
    tight_time_range,
)
from ..config import FOREST_DERIVED_CACHE_DIR, FOREST_SENTINEL_RASTER_CACHE_DIR
from .geometry import sample_geometries
from .models import DerivedPreview, Sample, SentinelDownload, SentinelSceneReview, SentinelSearch
from .storage_paths import forest_cache_url, to_storage_path


SEASON_PRESETS = {
    "early_summer": ("06-01", "07-15"),
    "late_summer": ("08-01", "09-15"),
    "full_snow_free": ("06-01", "09-15"),
}
S2_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12", "SCL", "dataMask"]
SENTINEL_VIEWS = ["rgb", "false_color", "ndvi", "nbr", "ndmi", "scl"]
SCL_CLOUD_CLASSES = {8, 9, 10}
SCL_SHADOW_CLASSES = {3}
SENTINEL_CLOUD_THRESHOLD = 0.10
SENTINEL_SHADOW_THRESHOLD = 0.35
SENTINEL_NODATA_THRESHOLD = 0.10
SENTINEL_DARK_FRACTION_THRESHOLD = 0.65
SENTINEL_DARK_BRIGHTNESS_THRESHOLD = 350.0
SENTINEL_DARK_P95_THRESHOLD = 600.0
AUTO_REVIEW_NOTES = "Auto-excluded by local Sentinel quality metrics."


@dataclass(frozen=True)
class SearchPeriod:
    key: str
    year: int
    start_mmdd: str
    end_mmdd: str

    @property
    def start(self) -> str:
        return f"{self.year}-{self.start_mmdd}T00:00:00Z"

    @property
    def end(self) -> str:
        return f"{self.year}-{self.end_mmdd}T23:59:59Z"

    @property
    def target_day_of_year(self) -> int:
        start = date.fromisoformat(f"{self.year}-{self.start_mmdd}")
        end = date.fromisoformat(f"{self.year}-{self.end_mmdd}")
        return int(round((start.timetuple().tm_yday + end.timetuple().tm_yday) / 2))


def item_day_distance(item: dict[str, Any], target_day_of_year: int) -> int | None:
    raw = item.get("datetime")
    if not raw:
        return None
    try:
        item_dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return abs(item_dt.timetuple().tm_yday - target_day_of_year)


def rank_item(item: dict[str, Any], period: SearchPeriod) -> dict[str, Any]:
    cloud = item.get("cloud_cover")
    cloud_cover = float(cloud) if cloud is not None else 1000.0
    day_distance = item_day_distance(item, period.target_day_of_year)
    score = cloud_cover + float(day_distance if day_distance is not None else 1000)
    return {
        **item,
        "day_of_year_distance": day_distance,
        "combined_score": round(score, 3),
    }


def periods_for_event_year(event_year: int, season_preset: str) -> list[SearchPeriod]:
    if season_preset not in SEASON_PRESETS:
        raise ValueError(f"Unknown season preset: {season_preset}")
    start_mmdd, end_mmdd = SEASON_PRESETS[season_preset]
    return [
        SearchPeriod("PRE", event_year - 1, start_mmdd, end_mmdd),
        SearchPeriod("EVENT", event_year, start_mmdd, end_mmdd),
        SearchPeriod("POST", event_year + 1, start_mmdd, end_mmdd),
    ]


def search_sentinel_for_sample(
    db: Session,
    sample: Sample,
    *,
    event_year: int,
    season_preset: str = "full_snow_free",
    max_cloud: float = 30.0,
    top_n: int = 10,
) -> dict[str, Any]:
    bbox = sample_geometries(sample.lon, sample.lat)["viewer_aoi_bbox"]
    periods = periods_for_event_year(event_year, season_preset)
    groups: list[dict[str, Any]] = []

    for period in periods:
        result = stac_search_s2(
            bbox=tuple(bbox),
            period=period,
            max_cloud=max_cloud,
            limit=max(top_n, 10),
        )
        ranked = sorted(
            (rank_item(item, period) for item in result["items"]),
            key=lambda item: item["combined_score"],
        )[:top_n]
        search_id = uuid.uuid4().hex
        db.add(
            SentinelSearch(
                search_id=search_id,
                sample_id=sample.sample_id,
                period=period.key,
                event_year=event_year,
                query_json=json.dumps(result.get("request"), ensure_ascii=False, sort_keys=True),
                results_json=json.dumps(ranked, ensure_ascii=False, sort_keys=True),
                cache_path=None,
            )
        )
        groups.append(
            {
                "period": period.key,
                "year": period.year,
                "start": period.start,
                "end": period.end,
                "target_day_of_year": period.target_day_of_year,
                "matched": result.get("matched"),
                "cache_status": result.get("cache_status"),
                "items": ranked,
            }
        )

    db.commit()
    return {
        "sample_id": sample.sample_id,
        "event_year": event_year,
        "season_preset": season_preset,
        "max_cloud": max_cloud,
        "top_n": top_n,
        "bbox": bbox,
        "groups": groups,
    }


def stable_id(payload: Any, length: int = 32) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:length]


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)[:160]


def pixel_size_for_bbox(bbox: list[float]) -> tuple[int, int]:
    west, south, east, north = bbox
    lat = (south + north) / 2
    meters_per_lon_degree = 111_320 * math.cos(math.radians(lat))
    meters_per_lat_degree = 110_574
    width_m = max(1.0, (east - west) * meters_per_lon_degree)
    height_m = max(1.0, (north - south) * meters_per_lat_degree)
    return (
        max(128, min(1024, int(round(width_m / 10)))),
        max(128, min(1024, int(round(height_m / 10)))),
    )


def cache_url(path: Path) -> str | None:
    return forest_cache_url(path)


def save_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_bytes(content)
    tmp_path.replace(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def stretch_channel(channel: np.ndarray, valid: np.ndarray) -> np.ndarray:
    data = channel[valid & np.isfinite(channel) & (channel > 0)]
    if data.size < 10:
        low, high = 0.0, 3000.0
    else:
        low, high = np.percentile(data, [2, 98])
        if high <= low:
            low, high = 0.0, max(float(data.max()), 1.0)
    return np.clip((channel - low) / max(high - low, 1.0) * 255.0, 0, 255).astype(np.uint8)


def rgba_composite(
    red: np.ndarray,
    green: np.ndarray,
    blue: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    rgba = np.zeros((*red.shape, 4), dtype=np.uint8)
    rgba[..., 0] = stretch_channel(red, valid)
    rgba[..., 1] = stretch_channel(green, valid)
    rgba[..., 2] = stretch_channel(blue, valid)
    rgba[..., 3] = np.where(valid, 255, 0).astype(np.uint8)
    return rgba


def normalized_index(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    return np.divide(a - b, a + b, out=np.zeros_like(a, dtype=np.float32), where=(a + b) != 0)


def colorize_index(
    values: np.ndarray,
    valid: np.ndarray,
    stops: list[tuple[float, tuple[int, int, int]]],
) -> np.ndarray:
    rgba = np.zeros((*values.shape, 4), dtype=np.uint8)
    clipped = np.clip(values, stops[0][0], stops[-1][0])
    for index in range(len(stops) - 1):
        left_value, left_color = stops[index]
        right_value, right_color = stops[index + 1]
        segment = (clipped >= left_value) & (clipped <= right_value)
        t = (clipped - left_value) / max(right_value - left_value, 1e-6)
        for channel in range(3):
            rgba[..., channel] = np.where(
                segment,
                (
                    left_color[channel]
                    + (right_color[channel] - left_color[channel]) * t
                ).astype(np.uint8),
                rgba[..., channel],
            )
    rgba[..., 3] = np.where(valid, 255, 0).astype(np.uint8)
    return rgba


def rgba_scl(scl: np.ndarray, valid: np.ndarray) -> np.ndarray:
    colors = {
        0: (0, 0, 0),
        1: (180, 0, 0),
        2: (90, 90, 90),
        3: (96, 54, 28),
        4: (40, 140, 67),
        5: (220, 196, 98),
        6: (50, 110, 190),
        7: (160, 160, 160),
        8: (225, 225, 225),
        9: (255, 255, 255),
        10: (170, 210, 255),
        11: (210, 245, 255),
    }
    rgba = np.zeros((*scl.shape, 4), dtype=np.uint8)
    for code, color in colors.items():
        mask = scl == code
        rgba[..., 0] = np.where(mask, color[0], rgba[..., 0])
        rgba[..., 1] = np.where(mask, color[1], rgba[..., 1])
        rgba[..., 2] = np.where(mask, color[2], rgba[..., 2])
    rgba[..., 3] = np.where(valid, 230, 0).astype(np.uint8)
    return rgba


def write_png(path: Path, rgba: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(path)


def preview_metadata(
    *,
    view: str,
    path: Path,
    bbox: list[float],
    width: int,
    height: int,
) -> dict[str, Any]:
    return {
        "view": view,
        "path": to_storage_path(path),
        "url": cache_url(path),
        "bbox": bbox,
        "width": width,
        "height": height,
    }


def derive_previews(
    db: Session,
    *,
    sample_id: str,
    download_id: str,
    tif_path: Path,
    bbox: list[float],
) -> list[dict[str, Any]]:
    with rasterio.open(tif_path) as dataset:
        arrays = {band: dataset.read(index + 1) for index, band in enumerate(S2_BANDS)}
        height = dataset.height
        width = dataset.width

    valid = arrays["dataMask"] > 0
    preview_dir = FOREST_DERIVED_CACHE_DIR / "sentinel" / sample_id / download_id
    views = {
        "rgb": rgba_composite(arrays["B04"], arrays["B03"], arrays["B02"], valid),
        "false_color": rgba_composite(arrays["B08"], arrays["B04"], arrays["B03"], valid),
        "ndvi": colorize_index(
            normalized_index(arrays["B08"], arrays["B04"]),
            valid,
            [
                (-1.0, (33, 42, 74)),
                (-0.1, (109, 86, 58)),
                (0.2, (219, 194, 117)),
                (0.45, (102, 158, 73)),
                (1.0, (25, 96, 50)),
            ],
        ),
        "nbr": colorize_index(
            normalized_index(arrays["B08"], arrays["B12"]),
            valid,
            [
                (-1.0, (70, 38, 26)),
                (-0.1, (169, 86, 56)),
                (0.25, (224, 194, 120)),
                (0.55, (92, 151, 86)),
                (1.0, (25, 92, 75)),
            ],
        ),
        "ndmi": colorize_index(
            normalized_index(arrays["B08"], arrays["B11"]),
            valid,
            [
                (-1.0, (95, 54, 31)),
                (-0.2, (190, 142, 78)),
                (0.0, (205, 188, 116)),
                (0.25, (91, 158, 169)),
                (1.0, (34, 82, 145)),
            ],
        ),
        "scl": rgba_scl(arrays["SCL"], valid),
    }

    metadata: list[dict[str, Any]] = []
    for view, rgba in views.items():
        png_path = preview_dir / f"{view}.png"
        json_path = preview_dir / f"{view}.json"
        if not png_path.exists() or not json_path.exists():
            write_png(png_path, rgba)
            write_json(
                json_path,
                preview_metadata(view=view, path=png_path, bbox=bbox, width=width, height=height),
            )

        preview = read_json(json_path) or preview_metadata(
            view=view,
            path=png_path,
            bbox=bbox,
            width=width,
            height=height,
        )
        preview_id = stable_id({"download_id": download_id, "view": view})
        existing = db.get(DerivedPreview, preview_id)
        storage_png_path = to_storage_path(png_path)
        storage_json_path = to_storage_path(json_path)
        if existing is None:
            db.add(
                DerivedPreview(
                    preview_id=preview_id,
                    download_id=download_id,
                    view=view,
                    png_path=storage_png_path,
                    metadata_path=storage_json_path,
                )
            )
        else:
            existing.png_path = storage_png_path
            existing.metadata_path = storage_json_path
        metadata.append(preview)
    return metadata


def cloud_metrics(tif_path: Path) -> dict[str, Any]:
    with rasterio.open(tif_path) as dataset:
        blue = dataset.read(1).astype(np.float32)
        green = dataset.read(2).astype(np.float32)
        red = dataset.read(3).astype(np.float32)
        nir = dataset.read(4).astype(np.float32)
        swir1 = dataset.read(5).astype(np.float32)
        swir2 = dataset.read(6).astype(np.float32)
        scl = dataset.read(7)
        data_mask = dataset.read(8) > 0
    total_count = int(data_mask.size)
    emptyish = data_mask & ((blue + green + red + nir + swir1 + swir2) <= 0)
    valid = data_mask & ~emptyish
    valid_count = int(valid.sum())
    nodata_count = int((~valid).sum())
    nodata_fraction = float(nodata_count / total_count) if total_count else 1.0
    if valid_count == 0:
        return {
            "cloud_fraction": 0.0,
            "shadow_fraction": 0.0,
            "nodata_fraction": 1.0,
            "dark_fraction": 1.0,
            "brightness_p50": 0.0,
            "brightness_p95": 0.0,
            "is_bad_cloud": False,
            "is_bad_shadow": False,
            "is_bad_nodata": True,
            "is_too_dark": True,
            "is_bad_quality": True,
            "quality_flags": ["NO_DATA_OR_BLACK_PIXELS", "TOO_DARK_OR_LOW_CONTRAST"],
        }

    cloud = np.isin(scl, list(SCL_CLOUD_CLASSES)) & valid
    shadow = np.isin(scl, list(SCL_SHADOW_CLASSES)) & valid
    cloud_fraction = float(cloud.sum() / valid_count)
    shadow_fraction = float(shadow.sum() / valid_count)
    brightness = (red + green + blue) / 3.0
    valid_brightness = brightness[valid & np.isfinite(brightness)]
    dark_fraction = (
        float((valid_brightness < SENTINEL_DARK_BRIGHTNESS_THRESHOLD).sum() / valid_brightness.size)
        if valid_brightness.size
        else 1.0
    )
    brightness_p50 = float(np.percentile(valid_brightness, 50)) if valid_brightness.size else 0.0
    brightness_p95 = float(np.percentile(valid_brightness, 95)) if valid_brightness.size else 0.0
    is_bad_cloud = cloud_fraction > SENTINEL_CLOUD_THRESHOLD
    is_bad_shadow = shadow_fraction > SENTINEL_SHADOW_THRESHOLD
    is_bad_nodata = nodata_fraction > SENTINEL_NODATA_THRESHOLD
    is_too_dark = (
        dark_fraction > SENTINEL_DARK_FRACTION_THRESHOLD
        or brightness_p95 < SENTINEL_DARK_P95_THRESHOLD
    )
    quality_flags = []
    if is_bad_nodata:
        quality_flags.append("NO_DATA_OR_BLACK_PIXELS")
    if is_bad_cloud:
        quality_flags.append("CLOUDS")
    if is_bad_shadow:
        quality_flags.append("CLOUD_SHADOW")
    if is_too_dark:
        quality_flags.append("TOO_DARK_OR_LOW_CONTRAST")
    return {
        "cloud_fraction": round(cloud_fraction, 4),
        "shadow_fraction": round(shadow_fraction, 4),
        "nodata_fraction": round(nodata_fraction, 4),
        "dark_fraction": round(dark_fraction, 4),
        "brightness_p50": round(brightness_p50, 2),
        "brightness_p95": round(brightness_p95, 2),
        "is_bad_cloud": is_bad_cloud,
        "is_bad_shadow": is_bad_shadow,
        "is_bad_nodata": is_bad_nodata,
        "is_too_dark": is_too_dark,
        "is_bad_quality": bool(quality_flags),
        "quality_flags": quality_flags,
    }


def auto_review_reason(metrics: dict[str, Any]) -> str | None:
    flags = set(metrics.get("quality_flags") or [])
    for reason in ("NO_DATA_OR_BLACK_PIXELS", "CLOUDS", "CLOUD_SHADOW", "TOO_DARK_OR_LOW_CONTRAST"):
        if reason in flags:
            return reason
    return None


def apply_auto_scene_review(db: Session, download_id: str, metrics: dict[str, Any]) -> None:
    reason_code = auto_review_reason(metrics)
    if reason_code is None:
        return
    existing = db.scalar(
        select(SentinelSceneReview).where(SentinelSceneReview.download_id == download_id)
    )
    if existing is not None:
        return
    db.add(
        SentinelSceneReview(
            download_id=download_id,
            is_excluded=True,
            reason_code=reason_code,
            reason_text=None,
            notes=AUTO_REVIEW_NOTES,
        )
    )


def download_sentinel_for_sample(
    db: Session,
    sample: Sample,
    *,
    item: dict[str, Any],
    period: str | None,
    max_cloud: float = 30.0,
) -> dict[str, Any]:
    if not item.get("id") or not item.get("datetime"):
        raise ValueError("A STAC item with id and datetime is required.")

    bbox = sample_geometries(sample.lon, sample.lat)["viewer_aoi_bbox"]
    width, height = pixel_size_for_bbox(bbox)
    download_id = stable_id(
        {
            "sample_id": sample.sample_id,
            "scene_id": item["id"],
            "bbox": bbox,
            "width": width,
            "height": height,
        }
    )
    scene_name = safe_name(item["id"])
    tif_path = FOREST_SENTINEL_RASTER_CACHE_DIR / sample.sample_id / f"{download_id}_{scene_name}.tif"
    metadata_path = tif_path.with_suffix(".json")
    cache_status = "hit"

    if not tif_path.exists() or not metadata_path.exists():
        time_from, time_to = tight_time_range(item["datetime"])
        payload = process_request(
            bbox=tuple(bbox),
            time_from=time_from,
            time_to=time_to,
            evalscript=S2_RAW_EVALSCRIPT,
            width=width,
            height=height,
            output_type="image/tiff",
            max_cloud=max_cloud,
        )
        content, content_type = run_process_api(payload, accept="image/tiff")
        save_bytes(tif_path, content)
        write_json(
            metadata_path,
            {
                "download_id": download_id,
                "sample_id": sample.sample_id,
                "period": period,
                "item": item,
                "bbox": bbox,
                "width": width,
                "height": height,
                "bands": S2_BANDS,
                "path": to_storage_path(tif_path),
                "content_type": content_type,
                "bytes": len(content),
                "request": payload,
            },
        )
        cache_status = "miss"

    metadata = read_json(metadata_path) or {}
    metrics = cloud_metrics(tif_path)
    previews = derive_previews(
        db,
        sample_id=sample.sample_id,
        download_id=download_id,
        tif_path=tif_path,
        bbox=bbox,
    )

    existing = db.get(SentinelDownload, download_id)
    storage_tif_path = to_storage_path(tif_path)
    if existing is None:
        existing = SentinelDownload(
            download_id=download_id,
            sample_id=sample.sample_id,
            scene_id=item["id"],
            period=period,
            local_path=storage_tif_path,
            bbox_json=json.dumps(bbox),
            bands_json=json.dumps(S2_BANDS),
            metadata_json=json.dumps(metadata, ensure_ascii=False, sort_keys=True),
        )
        db.add(existing)
    existing.period = period
    existing.local_path = storage_tif_path
    existing.bbox_json = json.dumps(bbox)
    existing.bands_json = json.dumps(S2_BANDS)
    existing.cloud_fraction = float(metrics["cloud_fraction"])
    existing.shadow_fraction = float(metrics["shadow_fraction"])
    existing.nodata_fraction = float(metrics["nodata_fraction"])
    existing.dark_fraction = float(metrics["dark_fraction"])
    existing.is_bad_cloud = bool(metrics["is_bad_cloud"])
    existing.is_bad_quality = bool(metrics["is_bad_quality"])
    existing.quality_flags_json = json.dumps(metrics["quality_flags"], ensure_ascii=False, sort_keys=True)
    existing.metadata_json = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    apply_auto_scene_review(db, download_id, metrics)
    db.commit()

    return {
        "download_id": download_id,
        "sample_id": sample.sample_id,
        "scene_id": item["id"],
        "period": period,
        "datetime": item.get("datetime"),
        "bbox": bbox,
        "width": width,
        "height": height,
        "bands": S2_BANDS,
        "local_path": storage_tif_path,
        "cache_status": cache_status,
        "cloud_fraction": metrics["cloud_fraction"],
        "shadow_fraction": metrics["shadow_fraction"],
        "nodata_fraction": metrics["nodata_fraction"],
        "dark_fraction": metrics["dark_fraction"],
        "is_bad_quality": metrics["is_bad_quality"],
        "quality_flags": metrics["quality_flags"],
        "is_bad_cloud": metrics["is_bad_cloud"],
        "previews": previews,
    }


def sentinel_review_payload(review: SentinelSceneReview | None) -> dict[str, Any]:
    return {
        "is_excluded": bool(review.is_excluded) if review is not None else False,
        "reason_code": review.reason_code if review is not None else None,
        "reason_text": review.reason_text if review is not None else None,
        "notes": review.notes if review is not None else None,
        "created_at": review.created_at.isoformat() if review is not None and review.created_at else None,
        "updated_at": review.updated_at.isoformat() if review is not None and review.updated_at else None,
    }


def json_or_list(value: str | None) -> list[Any]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def sentinel_download_payload(db: Session, download: SentinelDownload) -> dict[str, Any]:
    previews = list(
        db.scalars(select(DerivedPreview).where(DerivedPreview.download_id == download.download_id))
    )
    review = db.scalar(
        select(SentinelSceneReview).where(SentinelSceneReview.download_id == download.download_id)
    )
    metadata = json.loads(download.metadata_json)
    return {
        "download_id": download.download_id,
        "sample_id": download.sample_id,
        "scene_id": download.scene_id,
        "period": download.period,
        "datetime": metadata.get("item", {}).get("datetime"),
        "bbox": json.loads(download.bbox_json),
        "bands": json.loads(download.bands_json),
        "local_path": download.local_path,
        "cloud_fraction": download.cloud_fraction,
        "shadow_fraction": download.shadow_fraction,
        "nodata_fraction": download.nodata_fraction,
        "dark_fraction": download.dark_fraction,
        "is_bad_cloud": download.is_bad_cloud,
        "is_bad_quality": download.is_bad_quality,
        "quality_flags": json_or_list(download.quality_flags_json),
        "review": sentinel_review_payload(review),
        "previews": [
            {
                "view": preview.view,
                "path": preview.png_path,
                "url": forest_cache_url(preview.png_path),
                "bbox": json.loads(download.bbox_json),
            }
            for preview in previews
        ],
    }


def latest_sentinel_payload(db: Session, sample_id: str) -> dict[str, Any]:
    downloads = list(
        db.scalars(
            select(SentinelDownload)
            .where(SentinelDownload.sample_id == sample_id)
            .order_by(
                SentinelDownload.period,
                SentinelDownload.scene_id,
                SentinelDownload.created_at.desc(),
                SentinelDownload.download_id,
            )
        )
    )
    return {
        "downloads": [sentinel_download_payload(db, download) for download in downloads],
    }

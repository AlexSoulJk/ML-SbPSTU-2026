from __future__ import annotations

import csv
import hashlib
import json
import math
import uuid
from collections import Counter, defaultdict, deque
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.mask import mask as rio_mask
from rasterio.merge import merge as rio_merge
from rasterio.transform import array_bounds
from rasterio.warp import transform_geom
from scipy import ndimage
from shapely.geometry import shape
from shapely.ops import transform as shapely_transform
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..config import FOREST_EXPORTS_DIR, FOREST_HANSEN_CACHE_DIR
from .event_review_service import create_draft_review, require_actor_name
from .event_service import event_id_for
from .geometry import haversine_m, sample_geometries, transformers_for_lonlat
from .hansen_provider import (
    HANSEN_LAST_LOSS_YEAR,
    HANSEN_MAX_LOSS_CODE,
    HANSEN_VERSION,
    tile_id_for_lonlat,
)
from .hansen_service import (
    cache_url,
    loss_year_legend,
    pixel_area_ha,
    rgba_binary,
    rgba_loss_year,
    status_for_loss,
    write_png,
    write_raster,
)
from .models import (
    HansenAnalysis,
    HansenCandidate,
    HansenCandidateExclusion,
    HansenCandidateRun,
    HansenCandidateRunItem,
    HansenMask,
    HansenTile,
    Sample,
    SampleAssignment,
    SampleEvent,
    utc_now,
)
from .storage_paths import resolve_storage_path, to_storage_path


ALGORITHM_NAME = "HANSEN_CANDIDATE_MINING"
ALGORITHM_VERSION = "hansen_candidate_v1"
MIN_EVENT_YEAR = 2016
MIN_LOSS_CODE = MIN_EVENT_YEAR - 2000
MULTIPLE_EVENT_MIN_COUNT = 3
THRESHOLD_SWEEP = (0.5, 1.0, 2.5, 5.0, 10.0)
RUN_DECISIONS = {"ELIGIBLE", "SELECTED", "USER_REJECTED", "REJECTED"}
CANDIDATE_PREVIEW_SCHEMA_VERSION = 1
CANDIDATE_MASK_ANALYSIS_AREA_TYPE = "candidate_aoi"
CANDIDATE_MASK_TYPES = {"aoi_all_loss", "aoi_dominant_year"}

ProgressCallback = Callable[[str, str, int, int], None]
CancelCallback = Callable[[], bool]


@dataclass(frozen=True)
class CandidateMiningOptions:
    target_count: int = 100
    aoi_size_m: float = 2000.0
    safety_gap_m: float = 200.0
    min_dominant_event_aoi_percent: float = 1.0
    min_event_year: int = MIN_EVENT_YEAR
    scan_resolution_m: float = 300.0
    pool_multiplier: int = 3

    def validate(self) -> None:
        if not 1 <= self.target_count <= 1000:
            raise ValueError("target_count must be between 1 and 1000")
        if not 200 <= self.aoi_size_m <= 10000:
            raise ValueError("aoi_size_m must be between 200 and 10000 metres")
        if not 0 <= self.safety_gap_m <= 5000:
            raise ValueError("safety_gap_m must be between 0 and 5000 metres")
        if not 0 < self.min_dominant_event_aoi_percent <= 100:
            raise ValueError("min_dominant_event_aoi_percent must be in (0, 100]")
        if self.min_event_year < MIN_EVENT_YEAR:
            raise ValueError(f"min_event_year cannot be earlier than {MIN_EVENT_YEAR}")
        if self.min_event_year > HANSEN_LAST_LOSS_YEAR:
            raise ValueError(
                f"min_event_year cannot be later than Hansen coverage ({HANSEN_LAST_LOSS_YEAR})"
            )
        if not 60 <= self.scan_resolution_m <= 1000:
            raise ValueError("scan_resolution_m must be between 60 and 1000 metres")


@dataclass(frozen=True)
class TileInventoryItem:
    tile_id: str
    path: Path
    version: str


def stable_id(*parts: object) -> str:
    return hashlib.sha1("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_value(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


def _report(
    progress: ProgressCallback | None,
    stage: str,
    message: str,
    done: int,
    total: int,
) -> None:
    if progress:
        progress(stage, message, done, total)


def _raise_if_cancelled(should_cancel: CancelCallback | None) -> None:
    if should_cancel and should_cancel():
        raise RuntimeError("Hansen candidate mining cancelled")


def hansen_tile_inventory(db: Session) -> tuple[dict[str, TileInventoryItem], list[dict[str, Any]]]:
    rows = list(
        db.scalars(
            select(HansenTile)
            .where(HansenTile.layer == "lossyear")
            .order_by(HansenTile.tile_id, HansenTile.downloaded_at.desc(), HansenTile.id.desc())
        )
    )
    inventory: dict[str, TileInventoryItem] = {}
    checks: list[dict[str, Any]] = []
    for row in rows:
        if row.tile_id in inventory:
            continue
        registered_path = resolve_storage_path(row.local_path)
        path = registered_path
        check = {
            "tile_id": row.tile_id,
            "version": row.version,
            "path": to_storage_path(path) or str(path),
            "status": "ok",
            "error": None,
        }
        if "treecover2000" in {part.casefold() for part in path.parts}:
            canonical = (
                FOREST_HANSEN_CACHE_DIR
                / row.version
                / "lossyear"
                / f"{row.tile_id}.tif"
            )
            if canonical.exists():
                path = canonical
                check.update(
                    path=to_storage_path(path) or str(path),
                    registered_path=to_storage_path(registered_path) or str(registered_path),
                    status="recovered",
                    error="Registered lossyear path points to treecover2000; using canonical lossyear tile",
                )
            else:
                check.update(
                    status="invalid",
                    error="Registered lossyear path points to treecover2000 and canonical lossyear tile is missing",
                )
                checks.append(check)
                continue
        if not path.exists():
            check.update(status="missing", error="Tile file does not exist")
            checks.append(check)
            continue
        try:
            with rasterio.open(path) as dataset:
                if dataset.count < 1 or dataset.crs is None or dataset.transform is None:
                    raise ValueError("Raster has no band, CRS or transform")
                check.update(
                    width=dataset.width,
                    height=dataset.height,
                    crs=str(dataset.crs),
                    nodata=dataset.nodata,
                )
        except Exception as exc:
            check.update(status="invalid", error=str(exc)[:500])
            checks.append(check)
            continue
        inventory[row.tile_id] = TileInventoryItem(row.tile_id, path, row.version)
        checks.append(check)
    return inventory, checks


def _native_pixel_size_m(dataset: rasterio.io.DatasetReader) -> float:
    x_size = abs(float(dataset.transform.a))
    y_size = abs(float(dataset.transform.e))
    if dataset.crs and dataset.crs.is_geographic:
        center_lat = (float(dataset.bounds.bottom) + float(dataset.bounds.top)) / 2.0
        x_size *= 111_320.0 * max(0.1, math.cos(math.radians(center_lat)))
        y_size *= 110_574.0
    return max(0.01, min(x_size, y_size))


def _tile_proposals(
    item: TileInventoryItem,
    options: CandidateMiningOptions,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    with rasterio.open(item.path) as dataset:
        scale = max(1, int(math.ceil(options.scan_resolution_m / _native_pixel_size_m(dataset))))
        out_height = max(1, int(math.ceil(dataset.height / scale)))
        out_width = max(1, int(math.ceil(dataset.width / scale)))
        data = dataset.read(
            1,
            out_shape=(out_height, out_width),
            resampling=Resampling.nearest,
        ).astype(np.uint8, copy=False)
        invalid_codes = data[data > HANSEN_MAX_LOSS_CODE]
        if invalid_codes.size:
            raise ValueError(
                "Hansen lossyear raster contains code "
                f"{int(invalid_codes.max())}, but {HANSEN_VERSION} supports only "
                f"0..{HANSEN_MAX_LOSS_CODE}"
            )
        scan_transform = dataset.transform @ Affine.scale(
            dataset.width / out_width,
            dataset.height / out_height,
        )
        center_lat = (float(dataset.bounds.bottom) + float(dataset.bounds.top)) / 2.0
        coarse_pixel_area_ha = pixel_area_ha(scan_transform, dataset.crs, center_lat)

        proposals: list[dict[str, Any]] = []
        min_code = max(MIN_LOSS_CODE, options.min_event_year - 2000)
        structure = np.ones((3, 3), dtype=np.uint8)
        for code in sorted(
            int(value)
            for value in np.unique(data)
            if min_code <= int(value) <= HANSEN_MAX_LOSS_CODE
        ):
            binary = data == code
            labels, label_count = ndimage.label(binary, structure=structure)
            if label_count == 0:
                continue
            counts = np.bincount(labels.ravel())
            component_ids = np.flatnonzero(counts >= 2)
            component_ids = component_ids[component_ids > 0]
            if not component_ids.size:
                continue
            component_ids = component_ids[
                np.argsort(counts[component_ids], kind="stable")[::-1][:limit]
            ]
            centers = ndimage.center_of_mass(binary, labels, component_ids.tolist())
            for component_id, center in zip(component_ids.tolist(), centers, strict=True):
                if not np.isfinite(center[0]) or not np.isfinite(center[1]):
                    continue
                lon, lat = rasterio.transform.xy(
                    scan_transform,
                    float(center[0]),
                    float(center[1]),
                    offset="center",
                )
                component_key = f"{item.tile_id}:{2000 + code}:{component_id}"
                proposals.append(
                    {
                        "tile_id": item.tile_id,
                        "event_year": 2000 + code,
                        "lon": float(lon),
                        "lat": float(lat),
                        "component_key": component_key,
                        "component_area_ha": round(float(counts[component_id]) * coarse_pixel_area_ha, 4),
                        "coarse_pixel_count": int(counts[component_id]),
                        "scan_scale": scale,
                        "scan_resolution_m": round(_native_pixel_size_m(dataset) * scale, 2),
                    }
                )
        proposals.sort(
            key=lambda row: (
                -float(row["component_area_ha"]),
                int(row["event_year"]),
                str(row["component_key"]),
            )
        )
        return proposals[:limit]


def _required_tile_ids(bbox: list[float]) -> set[str]:
    left, bottom, right, top = [float(value) for value in bbox]
    epsilon = 1e-9
    lons = (left + epsilon, right - epsilon, (left + right) / 2.0)
    lats = (bottom + epsilon, top - epsilon, (bottom + top) / 2.0)
    return {tile_id_for_lonlat(lon, lat) for lon in lons for lat in lats}


def _aoi_histogram(
    lon: float,
    lat: float,
    aoi_size_m: float,
    inventory: dict[str, TileInventoryItem],
    min_event_year: int,
) -> dict[str, Any]:
    geometry_payload = sample_geometries(lon, lat, viewer_aoi_m=aoi_size_m)
    geometry = geometry_payload["viewer_aoi"]
    bbox = geometry_payload["viewer_aoi_bbox"]
    required_ids = _required_tile_ids(bbox)
    missing = sorted(required_ids - set(inventory))
    if missing:
        return {
            "status": "INCOMPLETE_HANSEN_COVERAGE",
            "missing_tile_ids": missing,
            "geometry": geometry,
            "bbox": bbox,
        }

    counts: Counter[int] = Counter()
    areas: Counter[int] = Counter()
    for tile_id in sorted(required_ids):
        item = inventory[tile_id]
        with rasterio.open(item.path) as dataset:
            raster_geometry = geometry
            if dataset.crs and str(dataset.crs).upper() not in {"EPSG:4326", "OGC:CRS84"}:
                raster_geometry = transform_geom("EPSG:4326", dataset.crs, geometry)
            data, transform = rio_mask(
                dataset,
                [raster_geometry],
                crop=True,
                filled=True,
                nodata=0,
            )
            layer = data[0].astype(np.uint8, copy=False)
            area_per_pixel = pixel_area_ha(transform, dataset.crs, lat)
            for code, count in zip(*np.unique(layer, return_counts=True), strict=True):
                year = 2000 + int(code)
                if int(code) <= 0 or int(code) > HANSEN_MAX_LOSS_CODE or year < min_event_year:
                    continue
                counts[year] += int(count)
                areas[year] += float(count) * area_per_pixel

    if not areas:
        return {
            "status": "NO_RECENT_LOSS",
            "geometry": geometry,
            "bbox": bbox,
            "source_tile_ids": sorted(required_ids),
        }
    histogram = [
        {
            "year": year,
            "pixels": counts[year],
            "loss_area_ha": round(areas[year], 4),
        }
        for year in sorted(areas)
    ]
    dominant_year = min(
        areas,
        key=lambda year: (-areas[year], year),
    )
    total_area = sum(areas.values())
    dominant_area = areas[dominant_year]
    aoi_area_ha = aoi_size_m * aoi_size_m / 10_000.0
    return {
        "status": "OK",
        "geometry": geometry,
        "bbox": bbox,
        "source_tile_ids": sorted(required_ids),
        "histogram": histogram,
        "event_years": sorted(areas),
        "pixel_count": sum(counts.values()),
        "dominant_event_year": dominant_year,
        "dominant_loss_area_ha": dominant_area,
        "dominant_loss_fraction": dominant_area / total_area if total_area else 0.0,
        "total_loss_area_ha": total_area,
        "post_2016_loss_aoi_percent": total_area / aoi_area_ha * 100.0,
        "dominant_event_aoi_percent": dominant_area / aoi_area_ha * 100.0,
        "has_multiple_events": len(areas) >= MULTIPLE_EVENT_MIN_COUNT,
    }


def _metric_geometry(geometry: dict[str, Any], lon: float, lat: float):
    to_metric, _to_lonlat, _epsg = transformers_for_lonlat(lon, lat)
    return shapely_transform(to_metric.transform, shape(geometry))


def _overlap(
    candidate_geometry: dict[str, Any],
    lon: float,
    lat: float,
    occupied: list[dict[str, Any]],
    safety_gap_m: float,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, float | None]:
    nearest: dict[str, Any] | None = None
    nearest_distance: float | None = None
    nearby: list[dict[str, Any]] = []
    for item in occupied:
        distance = haversine_m(lat, lon, float(item["lat"]), float(item["lon"]))
        if item["kind"] == "sample" and (nearest_distance is None or distance < nearest_distance):
            nearest, nearest_distance = item, distance
        if distance <= 6000.0 + safety_gap_m:
            nearby.append(item)
    if not nearby:
        return None, nearest, nearest_distance
    candidate_metric = _metric_geometry(candidate_geometry, lon, lat).buffer(safety_gap_m)
    to_metric, _to_lonlat, _epsg = transformers_for_lonlat(lon, lat)
    for item in nearby:
        other_metric = shapely_transform(to_metric.transform, shape(item["geometry"]))
        if candidate_metric.intersects(other_metric):
            return item, nearest, nearest_distance
    return None, nearest, nearest_distance


def _candidate_identity(
    primary_event_year: int,
    component_key: str,
    lon: float,
    lat: float,
    aoi_size_m: float,
) -> str:
    return stable_id(
        ALGORITHM_VERSION,
        HANSEN_VERSION,
        primary_event_year,
        component_key,
        round(lon, 5),
        round(lat, 5),
        round(aoi_size_m, 1),
    )


def _candidate_score(metrics: dict[str, Any], component_area_ha: float) -> float:
    return round(
        float(metrics["dominant_event_aoi_percent"]) * 10.0
        + float(metrics["post_2016_loss_aoi_percent"])
        + math.log1p(max(0.0, component_area_ha)),
        6,
    )


def _occupied_samples(db: Session, aoi_size_m: float) -> list[dict[str, Any]]:
    rows = list(db.scalars(select(Sample).order_by(Sample.sample_id)))
    return [
        {
            "kind": "sample",
            "id": row.sample_id,
            "lat": row.lat,
            "lon": row.lon,
            "geometry": sample_geometries(row.lon, row.lat, viewer_aoi_m=aoi_size_m)["viewer_aoi"],
        }
        for row in rows
    ]


def _active_exclusions(db: Session) -> list[dict[str, Any]]:
    rows = list(
        db.scalars(
            select(HansenCandidateExclusion)
            .where(HansenCandidateExclusion.is_active.is_(True))
            .order_by(HansenCandidateExclusion.exclusion_id)
        )
    )
    result = []
    for row in rows:
        geometry = _json_value(row.aoi_geometry_json, None)
        if not geometry:
            continue
        centroid = shape(geometry).centroid
        result.append(
            {
                "kind": "exclusion",
                "id": row.exclusion_id,
                "lat": centroid.y,
                "lon": centroid.x,
                "geometry": geometry,
            }
        )
    return result


def _round_robin_pool(
    by_tile: dict[str, list[dict[str, Any]]],
    occupied: list[dict[str, Any]],
    *,
    pool_target: int,
    safety_gap_m: float,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    queues = {
        tile_id: deque(
            sorted(rows, key=lambda row: (-row["candidate_score"], row["candidate_id"]))
        )
        for tile_id, rows in sorted(by_tile.items())
        if rows
    }
    accepted: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    dynamic_occupied = list(occupied)
    while queues and len(accepted) < pool_target:
        made_progress = False
        for tile_id in list(queues):
            queue = queues[tile_id]
            while queue:
                row = queue.popleft()
                collision, _nearest, _nearest_distance = _overlap(
                    row["aoi_geometry"],
                    row["lon"],
                    row["lat"],
                    dynamic_occupied,
                    safety_gap_m,
                )
                if collision is not None:
                    rejection_counts["CANDIDATE_OVERLAP"] += 1
                    continue
                accepted.append(row)
                dynamic_occupied.append(
                    {
                        "kind": "candidate",
                        "id": row["candidate_id"],
                        "lat": row["lat"],
                        "lon": row["lon"],
                        "geometry": row["aoi_geometry"],
                    }
                )
                made_progress = True
                break
            if not queue:
                queues.pop(tile_id, None)
            if len(accepted) >= pool_target:
                break
        if not made_progress:
            break
    return accepted, rejection_counts


def _upsert_candidate(db: Session, row: dict[str, Any]) -> HansenCandidate:
    candidate = db.get(HansenCandidate, row["candidate_id"])
    if candidate is None:
        candidate = HansenCandidate(candidate_id=row["candidate_id"])
        db.add(candidate)
    candidate.algorithm_version = ALGORITHM_VERSION
    candidate.hansen_dataset_version = HANSEN_VERSION
    candidate.lat = row["lat"]
    candidate.lon = row["lon"]
    candidate.aoi_size_m = row["aoi_size_m"]
    candidate.aoi_bbox_json = _json(row["aoi_bbox"])
    candidate.primary_event_year = row["primary_event_year"]
    candidate.event_years_json = _json(row["event_years"])
    candidate.has_multiple_events = row["has_multiple_events"]
    candidate.component_area_ha = row["component_area_ha"]
    candidate.dominant_loss_area_ha = row["dominant_loss_area_ha"]
    candidate.dominant_loss_fraction = row["dominant_loss_fraction"]
    candidate.total_loss_area_ha = row["total_loss_area_ha"]
    candidate.post_2016_loss_aoi_percent = row["post_2016_loss_aoi_percent"]
    candidate.dominant_event_aoi_percent = row["dominant_event_aoi_percent"]
    candidate.pixel_count = row["pixel_count"]
    candidate.source_tile_ids_json = _json(row["source_tile_ids"])
    candidate.component_key = row["component_key"]
    candidate.metadata_json = _json(row["metadata"])
    candidate.updated_at = utc_now()
    return candidate


def _write_artifacts(
    run: HansenCandidateRun,
    rows: list[dict[str, Any]],
    inventory_checks: list[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    artifact_dir = FOREST_EXPORTS_DIR / "hansen_candidate_mining" / run.run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = artifact_dir / "candidates.csv"
    inventory_path = artifact_dir / "tile_inventory.csv"
    summary_path = artifact_dir / "summary.json"
    candidate_rows = [
        {
            key: _json(value) if isinstance(value, (dict, list)) else value
            for key, value in row.items()
            if key != "aoi_geometry"
        }
        for row in rows
    ]
    with candidates_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = sorted({key for row in candidate_rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candidate_rows)
    with inventory_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = sorted({key for row in inventory_checks for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(inventory_checks)
    summary_path.write_text(_json(summary), encoding="utf-8")
    return to_storage_path(artifact_dir) or str(artifact_dir)


def mine_hansen_candidates(
    db: Session,
    options: CandidateMiningOptions,
    *,
    run_id: str | None = None,
    progress: ProgressCallback | None = None,
    should_cancel: CancelCallback | None = None,
) -> dict[str, Any]:
    options.validate()
    run = HansenCandidateRun(
        run_id=run_id or uuid.uuid4().hex,
        algorithm_name=ALGORITHM_NAME,
        algorithm_version=ALGORITHM_VERSION,
        hansen_dataset_version=HANSEN_VERSION,
        status="PLANNING",
        target_count=options.target_count,
        config_json=_json(asdict(options)),
    )
    db.add(run)
    db.commit()

    inventory, inventory_checks = hansen_tile_inventory(db)
    if not inventory:
        run.status = "FAILED"
        run.summary_json = _json({"error": "No valid local Hansen lossyear tiles"})
        db.commit()
        raise ValueError("No valid local Hansen lossyear tiles are registered in the database")

    occupied = _occupied_samples(db, options.aoi_size_m) + _active_exclusions(db)
    existing_candidate_ids = set(
        db.scalars(
            select(HansenCandidate.candidate_id).where(
                HansenCandidate.created_sample_id.is_not(None)
            )
        )
    )
    rejections: Counter[str] = Counter()
    sweep_counts: Counter[float] = Counter()
    by_tile: dict[str, list[dict[str, Any]]] = defaultdict(list)
    tile_count = len(inventory)
    per_tile_limit = max(20, min(100, math.ceil(options.target_count / tile_count) * 8))

    for tile_index, (tile_id, item) in enumerate(sorted(inventory.items()), start=1):
        _raise_if_cancelled(should_cancel)
        _report(progress, "scanning", f"Scanning Hansen tile {tile_id}", tile_index - 1, tile_count)
        try:
            proposals = _tile_proposals(item, options, limit=per_tile_limit)
        except Exception:
            rejections["TILE_SCAN_FAILED"] += 1
            continue
        for proposal in proposals:
            _raise_if_cancelled(should_cancel)
            metrics = _aoi_histogram(
                proposal["lon"],
                proposal["lat"],
                options.aoi_size_m,
                inventory,
                options.min_event_year,
            )
            if metrics["status"] != "OK":
                rejections[metrics["status"]] += 1
                continue
            for threshold in THRESHOLD_SWEEP:
                if metrics["dominant_event_aoi_percent"] >= threshold:
                    sweep_counts[threshold] += 1
            if metrics["dominant_event_aoi_percent"] < options.min_dominant_event_aoi_percent:
                rejections["BELOW_DOMINANT_THRESHOLD"] += 1
                continue
            collision, nearest, nearest_distance = _overlap(
                metrics["geometry"],
                proposal["lon"],
                proposal["lat"],
                occupied,
                options.safety_gap_m,
            )
            if collision is not None:
                rejections[
                    "ACTIVE_EXCLUSION" if collision["kind"] == "exclusion" else "EXISTING_SAMPLE_OVERLAP"
                ] += 1
                continue
            candidate_id = _candidate_identity(
                metrics["dominant_event_year"],
                proposal["component_key"],
                proposal["lon"],
                proposal["lat"],
                options.aoi_size_m,
            )
            if candidate_id in existing_candidate_ids:
                rejections["ALREADY_MATERIALIZED"] += 1
                continue
            row = {
                "candidate_id": candidate_id,
                "tile_id": tile_id,
                "lat": proposal["lat"],
                "lon": proposal["lon"],
                "aoi_size_m": options.aoi_size_m,
                "aoi_bbox": metrics["bbox"],
                "aoi_geometry": metrics["geometry"],
                "primary_event_year": metrics["dominant_event_year"],
                "event_years": metrics["event_years"],
                "has_multiple_events": metrics["has_multiple_events"],
                "component_area_ha": proposal["component_area_ha"],
                "dominant_loss_area_ha": round(metrics["dominant_loss_area_ha"], 4),
                "dominant_loss_fraction": round(metrics["dominant_loss_fraction"], 6),
                "total_loss_area_ha": round(metrics["total_loss_area_ha"], 4),
                "post_2016_loss_aoi_percent": round(metrics["post_2016_loss_aoi_percent"], 4),
                "dominant_event_aoi_percent": round(metrics["dominant_event_aoi_percent"], 4),
                "pixel_count": metrics["pixel_count"],
                "source_tile_ids": metrics["source_tile_ids"],
                "component_key": proposal["component_key"],
                "candidate_score": _candidate_score(metrics, proposal["component_area_ha"]),
                "nearest_sample_id": (
                    nearest["id"] if nearest is not None and nearest["kind"] == "sample" else None
                ),
                "nearest_sample_distance_m": round(nearest_distance, 2) if nearest_distance is not None else None,
                "metadata": {
                    "histogram": metrics["histogram"],
                    "discovery": proposal,
                    "min_event_year": options.min_event_year,
                },
            }
            by_tile[tile_id].append(row)
        _report(progress, "scanning", f"Scanned Hansen tile {tile_id}", tile_index, tile_count)

    pool_target = max(options.target_count, options.target_count * options.pool_multiplier)
    accepted, overlap_rejections = _round_robin_pool(
        by_tile,
        occupied,
        pool_target=pool_target,
        safety_gap_m=options.safety_gap_m,
    )
    rejections.update(overlap_rejections)
    for rank, row in enumerate(accepted, start=1):
        candidate = _upsert_candidate(db, row)
        db.flush()
        item_id = stable_id("candidate-run-item", run.run_id, candidate.candidate_id)
        run_item = db.get(HansenCandidateRunItem, item_id)
        if run_item is None:
            run_item = HansenCandidateRunItem(
                id=item_id,
                run_id=run.run_id,
                candidate_id=candidate.candidate_id,
            )
            db.add(run_item)
        run_item.decision = "SELECTED" if rank <= options.target_count else "ELIGIBLE"
        run_item.rank = rank
        run_item.candidate_score = row["candidate_score"]
        run_item.nearest_sample_id = row["nearest_sample_id"]
        run_item.nearest_sample_distance_m = row["nearest_sample_distance_m"]

    selected_count = min(options.target_count, len(accepted))
    summary = {
        "target_count": options.target_count,
        "selected_count": selected_count,
        "reserve_count": max(0, len(accepted) - selected_count),
        "valid_tiles": len(inventory),
        "invalid_tiles": sum(1 for row in inventory_checks if row["status"] != "ok"),
        "tile_candidate_counts": {key: len(value) for key, value in sorted(by_tile.items())},
        "threshold_sweep": {str(value): sweep_counts[value] for value in THRESHOLD_SWEEP},
        "rejections": dict(sorted(rejections.items())),
        "implementation": {
            "discovery": "downsampled connected components with exact full-resolution AOI metrics",
            "scan_resolution_m": options.scan_resolution_m,
        },
    }
    run.status = "PLANNED"
    run.selected_count = selected_count
    run.summary_json = _json(summary)
    run.artifact_path = _write_artifacts(run, accepted, inventory_checks, summary)
    run.updated_at = utc_now()
    db.commit()
    return candidate_run_payload(db, run.run_id)


def candidate_run_payload(db: Session, run_id: str) -> dict[str, Any]:
    run = db.get(HansenCandidateRun, run_id)
    if run is None:
        raise ValueError(f"Mining run not found: {run_id}")
    rows = db.execute(
        select(HansenCandidateRunItem, HansenCandidate)
        .join(HansenCandidate, HansenCandidate.candidate_id == HansenCandidateRunItem.candidate_id)
        .where(HansenCandidateRunItem.run_id == run_id)
        .order_by(HansenCandidateRunItem.rank, HansenCandidateRunItem.candidate_id)
    ).all()
    items = []
    for run_item, candidate in rows:
        items.append(
            {
                "candidate_id": candidate.candidate_id,
                "decision": run_item.decision,
                "rank": run_item.rank,
                "candidate_score": run_item.candidate_score,
                "rejection_reason": run_item.rejection_reason,
                "rejection_note": run_item.rejection_note,
                "lat": candidate.lat,
                "lon": candidate.lon,
                "aoi_size_m": candidate.aoi_size_m,
                "aoi_bbox": _json_value(candidate.aoi_bbox_json, []),
                "primary_event_year": candidate.primary_event_year,
                "event_years": _json_value(candidate.event_years_json, []),
                "has_multiple_events": candidate.has_multiple_events,
                "component_area_ha": candidate.component_area_ha,
                "dominant_loss_area_ha": candidate.dominant_loss_area_ha,
                "dominant_loss_fraction": candidate.dominant_loss_fraction,
                "total_loss_area_ha": candidate.total_loss_area_ha,
                "post_2016_loss_aoi_percent": candidate.post_2016_loss_aoi_percent,
                "dominant_event_aoi_percent": candidate.dominant_event_aoi_percent,
                "pixel_count": candidate.pixel_count,
                "source_tile_ids": _json_value(candidate.source_tile_ids_json, []),
                "created_sample_id": candidate.created_sample_id,
                "nearest_sample_id": run_item.nearest_sample_id,
                "nearest_sample_distance_m": run_item.nearest_sample_distance_m,
            }
        )
    artifacts = None
    if run.artifact_path:
        artifact_dir = resolve_storage_path(run.artifact_path)
        artifact_files = {
            "candidates_csv": artifact_dir / "candidates.csv",
            "tile_inventory_csv": artifact_dir / "tile_inventory.csv",
            "summary_json": artifact_dir / "summary.json",
        }
        available_artifacts = {
            name: to_storage_path(path)
            for name, path in artifact_files.items()
            if path.is_file()
        }
        artifacts = available_artifacts or None
    mask_status = candidate_mask_run_status(db, run_id)
    return {
        "run_id": run.run_id,
        "algorithm_name": run.algorithm_name,
        "algorithm_version": run.algorithm_version,
        "hansen_dataset_version": run.hansen_dataset_version,
        "status": run.status,
        "target_count": run.target_count,
        "selected_count": run.selected_count,
        "created_sample_count": run.created_sample_count,
        "hansen_masks": mask_status,
        "config": _json_value(run.config_json, {}),
        "summary": _json_value(run.summary_json, {}),
        "artifact_path": run.artifact_path,
        "artifacts": artifacts,
        "items": items,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "updated_at": run.updated_at.isoformat() if run.updated_at else None,
    }


def _candidate_lossyear_sources(
    db: Session,
    candidate: HansenCandidate,
) -> list[tuple[str, Path]]:
    tile_ids = [str(value) for value in _json_value(candidate.source_tile_ids_json, [])]
    if not tile_ids:
        tile_ids = sorted(_required_tile_ids(_json_value(candidate.aoi_bbox_json, [])))
    rows = list(
        db.scalars(
            select(HansenTile)
            .where(HansenTile.layer == "lossyear", HansenTile.tile_id.in_(tile_ids))
            .order_by(HansenTile.tile_id, HansenTile.downloaded_at.desc(), HansenTile.id.desc())
        )
    )
    rows_by_tile: dict[str, list[HansenTile]] = defaultdict(list)
    for row in rows:
        rows_by_tile[row.tile_id].append(row)

    sources: list[tuple[str, Path]] = []
    missing: list[str] = []
    for tile_id in tile_ids:
        candidates = sorted(
            rows_by_tile.get(tile_id, []),
            key=lambda row: row.version != candidate.hansen_dataset_version,
        )
        resolved = None
        for row in candidates:
            candidate_path = resolve_storage_path(row.local_path)
            if (
                "treecover2000" not in {part.casefold() for part in candidate_path.parts}
                and candidate_path.exists()
            ):
                resolved = candidate_path
                break
        if resolved is None:
            canonical = (
                FOREST_HANSEN_CACHE_DIR
                / candidate.hansen_dataset_version
                / "lossyear"
                / f"{tile_id}.tif"
            )
            if canonical.exists():
                resolved = canonical
        if resolved is None:
            missing.append(tile_id)
        else:
            sources.append((tile_id, resolved))
    if missing:
        raise FileNotFoundError(
            f"Hansen lossyear cache is missing candidate tile(s): {', '.join(missing)}"
        )
    if not sources:
        raise FileNotFoundError("Hansen candidate has no lossyear source tiles.")
    return sources


def _candidate_lossyear_raster(
    candidate: HansenCandidate,
    source_rows: list[tuple[str, Path]],
    *,
    min_event_year: int,
) -> tuple[np.ndarray, dict[str, Any], list[float]]:
    geometry_payload = sample_geometries(
        candidate.lon,
        candidate.lat,
        viewer_aoi_m=candidate.aoi_size_m,
    )
    geometry = geometry_payload["viewer_aoi"]
    requested_bbox = geometry_payload["viewer_aoi_bbox"]
    with ExitStack() as stack:
        datasets = [stack.enter_context(rasterio.open(path)) for _tile_id, path in source_rows]
        crs_values = {str(dataset.crs).upper() for dataset in datasets if dataset.crs}
        if crs_values not in ({"EPSG:4326"}, {"OGC:CRS84"}):
            raise ValueError(
                "Candidate Hansen masks expect all source tiles in EPSG:4326/CRS84."
            )
        source_profile = datasets[0].profile.copy()
        mosaic, transform = rio_merge(
            datasets,
            bounds=tuple(float(value) for value in requested_bbox),
            nodata=0,
            dtype="uint8",
            resampling=Resampling.nearest,
        )
    loss = mosaic[0].astype(np.uint8, copy=False)
    inside = geometry_mask(
        [geometry],
        out_shape=loss.shape,
        transform=transform,
        invert=True,
    )
    loss = np.where(inside, loss, 0).astype(np.uint8, copy=False)
    recent = (
        (loss > 0)
        & (loss <= HANSEN_MAX_LOSS_CODE)
        & ((loss.astype(np.int16) + 2000) >= min_event_year)
    )
    loss = np.where(recent, loss, 0).astype(np.uint8, copy=False)
    bbox = [float(value) for value in array_bounds(loss.shape[0], loss.shape[1], transform)]
    source_profile.update(
        transform=transform,
        width=loss.shape[1],
        height=loss.shape[0],
        count=1,
        dtype="uint8",
        nodata=0,
    )
    return loss, source_profile, bbox


def candidate_hansen_preview_payload(
    db: Session,
    candidate_id: str,
) -> dict[str, Any]:
    candidate = db.get(HansenCandidate, candidate_id)
    if candidate is None:
        raise ValueError(f"Hansen candidate not found: {candidate_id}")

    source_rows = _candidate_lossyear_sources(db, candidate)
    geometry_payload = sample_geometries(
        candidate.lon,
        candidate.lat,
        viewer_aoi_m=candidate.aoi_size_m,
    )
    requested_bbox = geometry_payload["viewer_aoi_bbox"]
    metadata = _json_value(candidate.metadata_json, {})
    min_event_year = int(metadata.get("min_event_year") or MIN_EVENT_YEAR)
    fingerprint = {
        "schema_version": CANDIDATE_PREVIEW_SCHEMA_VERSION,
        "candidate_id": candidate.candidate_id,
        "bbox": [round(float(value), 10) for value in requested_bbox],
        "min_event_year": min_event_year,
        "sources": [
            {
                "tile_id": tile_id,
                "path": to_storage_path(path) or str(path),
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for tile_id, path in source_rows
        ],
    }
    preview_dir = (
        FOREST_HANSEN_CACHE_DIR
        / "candidate_previews"
        / candidate.hansen_dataset_version
        / candidate.candidate_id
    )
    png_path = preview_dir / "all_loss.png"
    metadata_path = preview_dir / "all_loss.json"
    preview_metadata: dict[str, Any] | None = None
    cache_hit = False
    if png_path.exists() and metadata_path.exists():
        try:
            cached = json.loads(metadata_path.read_text(encoding="utf-8"))
            if cached.get("fingerprint") == fingerprint:
                preview_metadata = cached
                cache_hit = True
        except (OSError, ValueError, TypeError):
            preview_metadata = None

    if preview_metadata is None:
        loss, _profile, bbox = _candidate_lossyear_raster(
            candidate,
            source_rows,
            min_event_year=min_event_year,
        )

        preview_dir.mkdir(parents=True, exist_ok=True)
        temporary_png = png_path.with_name(f".{png_path.stem}.{uuid.uuid4().hex}.tmp.png")
        write_png(temporary_png, rgba_loss_year(loss))
        temporary_png.replace(png_path)
        preview_metadata = {
            "fingerprint": fingerprint,
            "bbox": [round(float(value), 10) for value in bbox],
            "legend": loss_year_legend(loss),
            "dimensions": [int(loss.shape[1]), int(loss.shape[0])],
        }
        temporary_metadata = metadata_path.with_name(
            f".{metadata_path.stem}.{uuid.uuid4().hex}.tmp.json"
        )
        temporary_metadata.write_text(
            json.dumps(preview_metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_metadata.replace(metadata_path)

    png_url = cache_url(png_path)
    if png_url is None:
        raise RuntimeError(f"Candidate Hansen preview is outside the forest cache: {png_path}")
    cache_version = max(source["mtime_ns"] for source in fingerprint["sources"])
    return {
        "preview_kind": "hansen_candidate_all_loss",
        "candidate_id": candidate.candidate_id,
        "primary_event_year": candidate.primary_event_year,
        "event_years": _json_value(candidate.event_years_json, []),
        "min_event_year": min_event_year,
        "png_url": f"{png_url}?v={cache_version}",
        "bbox": preview_metadata["bbox"],
        "legend": preview_metadata["legend"],
        "dimensions": preview_metadata["dimensions"],
        "cache_hit": cache_hit,
    }


def _candidate_mask_ready_sample_ids(
    db: Session,
    sample_ids: list[str],
) -> set[str]:
    if not sample_ids:
        return set()
    analyses = list(
        db.scalars(
            select(HansenAnalysis)
            .where(
                HansenAnalysis.sample_id.in_(sample_ids),
                HansenAnalysis.analysis_area_type == CANDIDATE_MASK_ANALYSIS_AREA_TYPE,
            )
            .order_by(
                HansenAnalysis.sample_id,
                HansenAnalysis.created_at.desc(),
                HansenAnalysis.analysis_id.desc(),
            )
        )
    )
    latest_by_sample: dict[str, HansenAnalysis] = {}
    for analysis in analyses:
        latest_by_sample.setdefault(analysis.sample_id, analysis)
    analysis_ids = [analysis.analysis_id for analysis in latest_by_sample.values()]
    masks = list(
        db.scalars(select(HansenMask).where(HansenMask.analysis_id.in_(analysis_ids)))
    ) if analysis_ids else []
    masks_by_analysis: dict[str, dict[str, HansenMask]] = defaultdict(dict)
    for mask in masks:
        masks_by_analysis[mask.analysis_id][mask.mask_type] = mask

    ready: set[str] = set()
    for sample_id, analysis in latest_by_sample.items():
        by_type = masks_by_analysis.get(analysis.analysis_id, {})
        if not CANDIDATE_MASK_TYPES.issubset(by_type):
            continue
        files_ready = True
        for mask_type in CANDIDATE_MASK_TYPES:
            tif_path = resolve_storage_path(by_type[mask_type].local_path)
            if not tif_path.exists() or not tif_path.with_suffix(".png").exists():
                files_ready = False
                break
        if files_ready:
            ready.add(sample_id)
    return ready


def candidate_mask_run_status(db: Session, run_id: str) -> dict[str, int]:
    candidates = list(
        db.scalars(
            select(HansenCandidate)
            .join(
                HansenCandidateRunItem,
                HansenCandidateRunItem.candidate_id == HansenCandidate.candidate_id,
            )
            .where(
                HansenCandidateRunItem.run_id == run_id,
                HansenCandidate.created_sample_id.is_not(None),
            )
        )
    )
    sample_ids = [
        candidate.created_sample_id
        for candidate in candidates
        if candidate.created_sample_id is not None
    ]
    ready = _candidate_mask_ready_sample_ids(db, sample_ids)
    return {
        "materialized": len(sample_ids),
        "ready": len(ready),
        "pending": len(sample_ids) - len(ready),
    }


def _write_candidate_masks(
    db: Session,
    candidate: HansenCandidate,
) -> HansenAnalysis:
    if not candidate.created_sample_id:
        raise ValueError(f"Candidate is not materialized: {candidate.candidate_id}")
    sample = db.get(Sample, candidate.created_sample_id)
    if sample is None:
        raise ValueError(f"Materialized sample is missing: {candidate.created_sample_id}")

    metadata = _json_value(candidate.metadata_json, {})
    min_event_year = int(metadata.get("min_event_year") or MIN_EVENT_YEAR)
    source_rows = _candidate_lossyear_sources(db, candidate)
    loss, profile, bbox = _candidate_lossyear_raster(
        candidate,
        source_rows,
        min_event_year=min_event_year,
    )
    histogram = metadata.get("histogram") or []
    if not histogram:
        area_ha = pixel_area_ha(profile["transform"], profile.get("crs"), candidate.lat)
        codes, counts = np.unique(
            loss[(loss > 0) & (loss <= HANSEN_MAX_LOSS_CODE)],
            return_counts=True,
        )
        histogram = [
            {
                "year": 2000 + int(code),
                "pixels": int(count),
                "loss_area_ha": round(float(count) * area_ha, 4),
            }
            for code, count in zip(codes.tolist(), counts.tolist(), strict=True)
        ]

    status = status_for_loss(
        float(candidate.total_loss_area_ha),
        float(candidate.dominant_loss_area_ha),
        float(candidate.dominant_loss_fraction),
    )
    analysis = HansenAnalysis(
        analysis_id=uuid.uuid4().hex,
        sample_id=sample.sample_id,
        analysis_area_type=CANDIDATE_MASK_ANALYSIS_AREA_TYPE,
        plot_size_m=float(candidate.aoi_size_m),
        treecover_threshold=None,
        event_year=int(candidate.primary_event_year),
        total_loss_area_ha=float(candidate.total_loss_area_ha),
        dominant_loss_area_ha=float(candidate.dominant_loss_area_ha),
        dominant_year_share=float(candidate.dominant_loss_fraction),
        status=status,
        histogram_json=_json(histogram),
    )
    db.add(analysis)

    dominant_code = int(candidate.primary_event_year) - 2000
    dominant = loss == dominant_code
    mask_dir = FOREST_HANSEN_CACHE_DIR / "masks" / sample.sample_id / analysis.analysis_id
    mask_specs = (
        ("aoi_all_loss", loss, rgba_loss_year(loss)),
        ("aoi_dominant_year", dominant.astype(np.uint8), rgba_binary(dominant, (220, 70, 45))),
    )
    for mask_type, raster_data, png_data in mask_specs:
        tif_path = mask_dir / f"{mask_type}.tif"
        write_raster(tif_path, raster_data, profile)
        write_png(tif_path.with_suffix(".png"), png_data)
        db.add(
            HansenMask(
                mask_id=uuid.uuid4().hex,
                sample_id=sample.sample_id,
                analysis_id=analysis.analysis_id,
                mask_type=mask_type,
                local_path=to_storage_path(tif_path) or str(tif_path),
                bbox_json=_json(bbox),
            )
        )
    return analysis


def generate_candidate_masks_for_run(
    db: Session,
    run_id: str,
    *,
    execution_id: str | None = None,
    progress: ProgressCallback | None = None,
    should_cancel: CancelCallback | None = None,
) -> dict[str, Any]:
    run = db.get(HansenCandidateRun, run_id)
    if run is None:
        raise ValueError(f"Mining run not found: {run_id}")
    candidates = list(
        db.scalars(
            select(HansenCandidate)
            .join(
                HansenCandidateRunItem,
                HansenCandidateRunItem.candidate_id == HansenCandidate.candidate_id,
            )
            .where(
                HansenCandidateRunItem.run_id == run_id,
                HansenCandidate.created_sample_id.is_not(None),
            )
            .order_by(HansenCandidate.primary_event_year, HansenCandidate.candidate_id)
        )
    )
    if not candidates:
        raise ValueError("Mining run has no materialized candidates")

    sample_ids = [
        candidate.created_sample_id
        for candidate in candidates
        if candidate.created_sample_id is not None
    ]
    ready_ids = _candidate_mask_ready_sample_ids(db, sample_ids)
    report_rows: list[dict[str, Any]] = []
    created = 0
    reused = 0
    failed = 0
    cancelled = False
    total = len(candidates)
    for index, candidate in enumerate(candidates, start=1):
        if should_cancel and should_cancel():
            cancelled = True
            break
        sample_id = candidate.created_sample_id or ""
        if progress:
            progress(
                "candidate_masks",
                f"Hansen AOI masks {index}/{total}: {sample_id[:8]}",
                index - 1,
                total,
            )
        if sample_id in ready_ids:
            reused += 1
            report_rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "sample_id": sample_id,
                    "status": "reused",
                    "error": "",
                }
            )
        else:
            try:
                analysis = _write_candidate_masks(db, candidate)
                db.commit()
                created += 1
                report_rows.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "sample_id": sample_id,
                        "analysis_id": analysis.analysis_id,
                        "status": "created",
                        "error": "",
                    }
                )
            except Exception as exc:
                db.rollback()
                failed += 1
                report_rows.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "sample_id": sample_id,
                        "status": "failed",
                        "error": str(exc)[:500],
                    }
                )
        if progress:
            progress(
                "candidate_masks",
                f"Hansen AOI masks: {index}/{total}; failed {failed}",
                index,
                total,
            )

    artifact_id = execution_id or uuid.uuid4().hex
    artifact_dir = FOREST_EXPORTS_DIR / "hansen_candidate_masks" / artifact_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    csv_path = artifact_dir / "mask_generation.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = ["candidate_id", "sample_id", "analysis_id", "status", "error"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report_rows)
    summary = {
        "run_id": run_id,
        "status": "CANCELLED" if cancelled else "COMPLETED_WITH_ERRORS" if failed else "COMPLETED",
        "total": total,
        "processed": created + reused + failed,
        "created": created,
        "reused": reused,
        "failed": failed,
        "events_modified": 0,
        "reviews_modified": 0,
        "sentinel_links_modified": 0,
        "csv_path": to_storage_path(csv_path) or str(csv_path),
    }
    summary_path = artifact_dir / "summary.json"
    summary["summary_json"] = to_storage_path(summary_path) or str(summary_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def list_candidate_runs(db: Session, limit: int = 25) -> list[dict[str, Any]]:
    runs = list(
        db.scalars(
            select(HansenCandidateRun)
            .order_by(HansenCandidateRun.created_at.desc())
            .limit(limit)
        )
    )
    return [candidate_run_payload(db, run.run_id) for run in runs]


def _refresh_run_selection_counts(db: Session, run: HansenCandidateRun) -> None:
    db.flush()
    selected = int(
        db.scalar(
            select(func.count()).select_from(HansenCandidateRunItem).where(
                HansenCandidateRunItem.run_id == run.run_id,
                HansenCandidateRunItem.decision == "SELECTED",
            )
        ) or 0
    )
    reserve = int(
        db.scalar(
            select(func.count()).select_from(HansenCandidateRunItem).where(
                HansenCandidateRunItem.run_id == run.run_id,
                HansenCandidateRunItem.decision == "ELIGIBLE",
            )
        ) or 0
    )
    run.selected_count = selected
    summary = _json_value(run.summary_json, {})
    summary["selected_count"] = selected
    summary["reserve_count"] = reserve
    run.summary_json = _json(summary)
    run.updated_at = utc_now()


def reject_candidates(
    db: Session,
    run_id: str,
    candidate_ids: list[str],
    *,
    reason_code: str,
    reason_text: str | None,
    actor_name: str | None,
) -> dict[str, Any]:
    actor = require_actor_name(actor_name)
    if not reason_code.strip():
        raise ValueError("reason_code is required")
    unique_ids = list(dict.fromkeys(candidate_ids))
    for candidate_id in unique_ids:
        candidate = db.get(HansenCandidate, candidate_id)
        if candidate is None:
            raise ValueError(f"Candidate not found: {candidate_id}")
        run_item = db.scalar(
            select(HansenCandidateRunItem).where(
                HansenCandidateRunItem.run_id == run_id,
                HansenCandidateRunItem.candidate_id == candidate_id,
            )
        )
        if run_item is None:
            raise ValueError(f"Candidate {candidate_id} is not part of run {run_id}")
        active = db.scalar(
            select(HansenCandidateExclusion).where(
                HansenCandidateExclusion.candidate_id == candidate_id,
                HansenCandidateExclusion.is_active.is_(True),
            )
        )
        if active is None:
            geometry = sample_geometries(
                candidate.lon,
                candidate.lat,
                viewer_aoi_m=candidate.aoi_size_m,
            )["viewer_aoi"]
            db.add(
                HansenCandidateExclusion(
                    exclusion_id=uuid.uuid4().hex,
                    candidate_id=candidate_id,
                    aoi_bbox_json=candidate.aoi_bbox_json,
                    aoi_geometry_json=_json(geometry),
                    reason_code=reason_code.strip().upper(),
                    reason_text=(reason_text or "").strip() or None,
                    actor_name=actor,
                )
            )
        run_item.decision = "USER_REJECTED"
        run_item.rejection_reason = reason_code.strip().upper()
        run_item.rejection_note = (reason_text or "").strip() or None
    run = db.get(HansenCandidateRun, run_id)
    if run is None:
        raise ValueError(f"Mining run not found: {run_id}")
    _refresh_run_selection_counts(db, run)
    db.commit()
    return candidate_run_payload(db, run_id)


def fill_candidate_run(db: Session, run_id: str) -> dict[str, Any]:
    run = db.get(HansenCandidateRun, run_id)
    if run is None:
        raise ValueError(f"Mining run not found: {run_id}")
    selected = int(
        db.scalar(
            select(func.count()).select_from(HansenCandidateRunItem).where(
                HansenCandidateRunItem.run_id == run_id,
                HansenCandidateRunItem.decision == "SELECTED",
            )
        ) or 0
    )
    needed = max(0, run.target_count - selected)
    reserve = list(
        db.scalars(
            select(HansenCandidateRunItem)
            .where(
                HansenCandidateRunItem.run_id == run_id,
                HansenCandidateRunItem.decision == "ELIGIBLE",
            )
            .order_by(HansenCandidateRunItem.rank, HansenCandidateRunItem.candidate_id)
            .limit(needed)
        )
    )
    for item in reserve:
        item.decision = "SELECTED"
    _refresh_run_selection_counts(db, run)
    db.commit()
    return candidate_run_payload(db, run_id)


def restore_exclusion(db: Session, exclusion_id: str) -> dict[str, Any]:
    exclusion = db.get(HansenCandidateExclusion, exclusion_id)
    if exclusion is None:
        raise ValueError(f"Exclusion not found: {exclusion_id}")
    exclusion.is_active = False
    exclusion.restored_at = utc_now()
    db.commit()
    return {"exclusion_id": exclusion_id, "is_active": False}


def _sample_id_for(candidate_id: str) -> str:
    return stable_id("hansen-mined-sample", candidate_id)


def materialize_candidates(
    db: Session,
    run_id: str,
    candidate_ids: list[str],
) -> dict[str, Any]:
    run = db.get(HansenCandidateRun, run_id)
    if run is None:
        raise ValueError(f"Mining run not found: {run_id}")
    unique_ids = list(dict.fromkeys(candidate_ids))
    created = 0
    existing = 0
    created_sample_ids: list[str] = []
    for candidate_id in unique_ids:
        candidate = db.get(HansenCandidate, candidate_id)
        if candidate is None:
            raise ValueError(f"Candidate not found: {candidate_id}")
        item = db.scalar(
            select(HansenCandidateRunItem).where(
                HansenCandidateRunItem.run_id == run_id,
                HansenCandidateRunItem.candidate_id == candidate_id,
            )
        )
        if item is None or item.decision not in {"SELECTED", "ELIGIBLE"}:
            raise ValueError(f"Candidate {candidate_id} is not eligible in run {run_id}")
        if candidate.created_sample_id:
            existing += 1
            created_sample_ids.append(candidate.created_sample_id)
            continue

        sample_id = _sample_id_for(candidate_id)
        sample = db.get(Sample, sample_id)
        metadata = _json_value(candidate.metadata_json, {})
        event_years = _json_value(candidate.event_years_json, [])
        raw_properties = {
            "origin": "HANSEN_MINING",
            "candidate_id": candidate_id,
            "mining_run_id": run_id,
            "algorithm_version": candidate.algorithm_version,
            "hansen_dataset_version": candidate.hansen_dataset_version,
            "aoi_bbox": _json_value(candidate.aoi_bbox_json, []),
            "event_years": event_years,
            "candidate_metrics": {
                "component_area_ha": candidate.component_area_ha,
                "dominant_loss_area_ha": candidate.dominant_loss_area_ha,
                "total_loss_area_ha": candidate.total_loss_area_ha,
                "post_2016_loss_aoi_percent": candidate.post_2016_loss_aoi_percent,
                "dominant_event_aoi_percent": candidate.dominant_event_aoi_percent,
            },
        }
        if sample is None:
            sample = Sample(
                sample_id=sample_id,
                source_file=ALGORITHM_VERSION,
                source_format="HANSEN_MINING",
                source_id=candidate_id,
                display_name=f"Hansen {candidate.primary_event_year} · {candidate_id[:8]}",
                lat=candidate.lat,
                lon=candidate.lon,
                source_event_year=candidate.primary_event_year,
                driver_primary=None,
                confidence_primary=None,
                has_multiple_events=candidate.has_multiple_events,
                raw_properties_json=_json(raw_properties),
            )
            db.add(sample)
            db.flush()
            db.add(
                SampleAssignment(
                    sample_id=sample_id,
                    assignment_type="hansen_mining_run",
                    assignment_name=run_id,
                )
            )
            histogram_by_year = {
                int(row["year"]): row
                for row in metadata.get("histogram", [])
                if row.get("year") is not None
            }
            for year in event_years:
                year = int(year)
                histogram_row = histogram_by_year.get(year, {})
                area = float(histogram_row.get("loss_area_ha") or 0.0)
                event = SampleEvent(
                    event_id=event_id_for(sample_id, "HANSEN", year),
                    sample_id=sample_id,
                    event_year=year,
                    source="HANSEN",
                    loss_area_ha=area,
                    loss_share=(area / candidate.total_loss_area_ha if candidate.total_loss_area_ha else 0.0),
                    is_dominant=year == candidate.primary_event_year,
                    metadata_json=_json(
                        {
                            "candidate_id": candidate_id,
                            "mining_run_id": run_id,
                            "pixels": histogram_row.get("pixels"),
                        }
                    ),
                )
                db.add(event)
                db.flush()
                create_draft_review(db, event.event_id)
            created += 1
        else:
            existing += 1
        candidate.created_sample_id = sample_id
        candidate.updated_at = utc_now()
        item.decision = "SELECTED"
        created_sample_ids.append(sample_id)

    db.flush()
    run.created_sample_count = int(
        db.scalar(
            select(func.count(HansenCandidate.candidate_id))
            .join(
                HansenCandidateRunItem,
                HansenCandidateRunItem.candidate_id == HansenCandidate.candidate_id,
            )
            .where(
                HansenCandidateRunItem.run_id == run_id,
                HansenCandidate.created_sample_id.is_not(None),
            )
        ) or 0
    )
    run.status = "APPLIED" if run.created_sample_count >= run.selected_count else "PARTIAL"
    run.updated_at = utc_now()
    db.commit()
    return {
        "run": candidate_run_payload(db, run_id),
        "created": created,
        "existing": existing,
        "sample_ids": created_sample_ids,
    }

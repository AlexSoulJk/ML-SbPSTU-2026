from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .cdse_client import (
    best_s2_observations,
    cdse_uses_env_proxy,
    configured_proxy_keys,
    fetch_s2_preview,
    fetch_s2_source,
    preview_filename,
    source_filename,
)
from .config import (
    DEFAULT_CONTEXT_M,
    DEFAULT_KML_PATH,
    DEFAULT_MAX_CLOUD,
    DEFAULT_MONTHS,
    DEFAULT_SEGMENT_LENGTH_M,
    DEFAULT_YEARS,
    FALLBACK_KML_PATH,
    FOREST_CACHE_DIR,
    FRONTEND_DIR,
    PROCESSED_DIR,
    PREVIEW_CACHE_DIR,
    S2_CACHE_DIR,
    S2_RESOLUTION_M,
    ensure_directories,
    load_dotenv,
)
from .forest.database import init_forest_database
from .forest.routes import router as forest_router
from .geometry import geometry_payload, stable_geometry_id
from .kml_loader import kml_document_to_api, parse_kml_file, parse_kml_text


app = FastAPI(title="Satellite Pipeline Explorer", version="0.1.0")
S2_PREVIEW_VIEWS = ["rgb", "false_color", "ndvi", "ndmi"]


class KmlParseRequest(BaseModel):
    filename: str = "uploaded.kml"
    text: str


class GeometryRequest(BaseModel):
    geometry_id: str
    coordinates: list[tuple[float, float]]
    context_m: float = DEFAULT_CONTEXT_M
    segment_length_m: float = DEFAULT_SEGMENT_LENGTH_M
    skeleton_pixel_size_m: float = 25


class ObservationRequest(BaseModel):
    bbox: tuple[float, float, float, float]
    years: list[int] = Field(default_factory=lambda: DEFAULT_YEARS.copy())
    months: list[int] = Field(default_factory=lambda: DEFAULT_MONTHS.copy())
    max_cloud: float = DEFAULT_MAX_CLOUD


class PreviewRequest(BaseModel):
    bbox: tuple[float, float, float, float]
    item: dict[str, Any]
    views: list[str] = Field(default_factory=lambda: ["rgb", "false_color", "ndvi", "ndmi"])
    max_cloud: float = DEFAULT_MAX_CLOUD


def copy_default_kml_if_needed() -> None:
    if DEFAULT_KML_PATH.exists() or not FALLBACK_KML_PATH.exists():
        return
    DEFAULT_KML_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(FALLBACK_KML_PATH, DEFAULT_KML_PATH)


@app.on_event("startup")
def startup() -> None:
    ensure_directories()
    load_dotenv()
    copy_default_kml_if_needed()
    init_forest_database()


app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")
app.mount("/cache/previews", StaticFiles(directory=PREVIEW_CACHE_DIR), name="previews")
app.mount("/cache/forest", StaticFiles(directory=FOREST_CACHE_DIR), name="forest-cache")
app.include_router(forest_router)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "default_kml_exists": DEFAULT_KML_PATH.exists(),
        "tool": "Satellite Pipeline Explorer",
    }


@app.get("/api/kml/default")
def default_kml() -> dict[str, Any]:
    if not DEFAULT_KML_PATH.exists():
        raise HTTPException(status_code=404, detail="Default KML file was not found.")
    document = parse_kml_file(DEFAULT_KML_PATH)
    payload = kml_document_to_api(document)
    payload["source_path"] = str(DEFAULT_KML_PATH)
    return payload


@app.post("/api/kml/parse")
def parse_kml(request: KmlParseRequest) -> dict[str, Any]:
    try:
        document = parse_kml_text(request.text)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    payload = kml_document_to_api(document)
    payload["source_filename"] = request.filename
    return payload


@app.post("/api/geometry/segments")
def segments(request: GeometryRequest) -> dict[str, Any]:
    try:
        payload = geometry_payload(
            kml_id=request.geometry_id or stable_geometry_id(request.coordinates),
            boundary_coordinates=request.coordinates,
            context_m=request.context_m,
            segment_length_m=request.segment_length_m,
            skeleton_pixel_size_m=request.skeleton_pixel_size_m,
        )
        processed_dir = PROCESSED_DIR / payload["kml_id"]
        processed_dir.mkdir(parents=True, exist_ok=True)
        (processed_dir / "geometry.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return payload


@app.post("/api/sentinel2/observations")
def sentinel2_observations(request: ObservationRequest) -> dict[str, Any]:
    try:
        observations = best_s2_observations(
            bbox=request.bbox,
            years=request.years,
            months=request.months,
            max_cloud=request.max_cloud,
        )
        for observation in observations:
            item = observation.get("best_item")
            if item:
                observation["preview_cache"] = sentinel2_preview_cache_status(
                    request.bbox,
                    item,
                    S2_PREVIEW_VIEWS,
                )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"bbox": request.bbox, "observations": observations}


def pixel_size_for_bbox(bbox: tuple[float, float, float, float]) -> tuple[int, int]:
    west, south, east, north = bbox
    lat = (south + north) / 2
    meters_per_lon_degree = 111_320 * math.cos(math.radians(lat))
    meters_per_lat_degree = 110_574
    width_m = max(1.0, (east - west) * meters_per_lon_degree)
    height_m = max(1.0, (north - south) * meters_per_lat_degree)
    return (
        max(128, min(1024, int(round(width_m / S2_RESOLUTION_M)))),
        max(128, min(1024, int(round(height_m / S2_RESOLUTION_M)))),
    )


def sentinel2_preview_cache_status(
    bbox: tuple[float, float, float, float],
    item: dict[str, Any],
    views: list[str],
) -> dict[str, Any]:
    width, height = pixel_size_for_bbox(bbox)
    item_id = item.get("id")
    if not item_id:
        return {
            "width": width,
            "height": height,
            "source_cached": False,
            "cached_views": [],
            "all_views_cached": False,
        }

    source_path = S2_CACHE_DIR / source_filename(item_id, bbox, width, height)
    cached_views = []
    for view in views:
        preview_path = PREVIEW_CACHE_DIR / preview_filename(item_id, view, bbox, width, height)
        if preview_path.exists() and preview_path.with_suffix(".json").exists():
            cached_views.append(view)

    return {
        "width": width,
        "height": height,
        "source_cached": source_path.exists() and source_path.with_suffix(".json").exists(),
        "cached_views": cached_views,
        "all_views_cached": all(view in cached_views for view in views),
    }


@app.post("/api/sentinel2/previews")
def sentinel2_previews(request: PreviewRequest) -> dict[str, Any]:
    width, height = pixel_size_for_bbox(request.bbox)
    try:
        source = fetch_s2_source(
            bbox=request.bbox,
            item=request.item,
            width=width,
            height=height,
            max_cloud=request.max_cloud,
        )
        previews = [
            fetch_s2_preview(
                bbox=request.bbox,
                item=request.item,
                view=view,
                width=width,
                height=height,
                max_cloud=request.max_cloud,
            )
            for view in request.views
        ]
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {
        "bbox": request.bbox,
        "width": width,
        "height": height,
        "source": source,
        "previews": previews,
        "preview_cache": sentinel2_preview_cache_status(request.bbox, request.item, request.views),
    }


@app.get("/api/cache/summary")
def cache_summary() -> dict[str, Any]:
    preview_count = len(list(PREVIEW_CACHE_DIR.glob("*.png")))
    source_count = len(list((PREVIEW_CACHE_DIR.parent / "sentinel2").glob("*.tif")))
    stac_count = len(list((PREVIEW_CACHE_DIR.parent / "stac").glob("*.json")))
    return {
        "stac_requests": stac_count,
        "source_tiffs": source_count,
        "preview_pngs": preview_count,
    }


@app.get("/api/debug/config")
def debug_config() -> dict[str, Any]:
    return {
        "frontend": str(FRONTEND_DIR),
        "default_kml": str(DEFAULT_KML_PATH),
        "preview_cache": str(PREVIEW_CACHE_DIR),
        "cdse_use_env_proxy": cdse_uses_env_proxy(),
        "proxy_env_keys": configured_proxy_keys(),
    }

from __future__ import annotations

import calendar
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pyproj import Transformer
import requests

from .config import (
    CRS84,
    PREVIEW_CACHE_DIR,
    PROCESS_URL,
    S2_CACHE_DIR,
    STAC_CACHE_DIR,
    STAC_SEARCH_URL,
    TOKEN_URL,
)


ENV_TRUE_VALUES = {"1", "true", "yes", "on"}
PROXY_ENV_KEYS = (
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "ALL_PROXY",
    "https_proxy",
    "http_proxy",
    "all_proxy",
)
WEB_MERCATOR_CRS = "http://www.opengis.net/def/crs/EPSG/0/3857"
TO_WEB_MERCATOR = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

S2_RAW_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B02", "B03", "B04", "B08", "B11", "B12", "SCL", "dataMask"],
      units: "DN"
    }],
    output: {
      id: "default",
      bands: 8,
      sampleType: "UINT16"
    }
  };
}

function evaluatePixel(sample) {
  return [
    sample.B02,
    sample.B03,
    sample.B04,
    sample.B08,
    sample.B11,
    sample.B12,
    sample.SCL,
    sample.dataMask
  ];
}
""".strip()

S2_VIEW_EVALSCRIPTS = {
    "rgb": """
//VERSION=3
function setup() {
  return {
    input: ["B02", "B03", "B04", "dataMask"],
    output: { bands: 4, sampleType: "AUTO" }
  };
}

function stretch(value) {
  return Math.min(1, Math.max(0, value * 2.8));
}

function evaluatePixel(sample) {
  return [
    stretch(sample.B04),
    stretch(sample.B03),
    stretch(sample.B02),
    sample.dataMask
  ];
}
""".strip(),
    "false_color": """
//VERSION=3
function setup() {
  return {
    input: ["B03", "B04", "B08", "dataMask"],
    output: { bands: 4, sampleType: "AUTO" }
  };
}

function stretch(value) {
  return Math.min(1, Math.max(0, value * 2.8));
}

function evaluatePixel(sample) {
  return [
    stretch(sample.B08),
    stretch(sample.B04),
    stretch(sample.B03),
    sample.dataMask
  ];
}
""".strip(),
    "ndvi": """
//VERSION=3
function setup() {
  return {
    input: ["B04", "B08", "dataMask"],
    output: { bands: 4, sampleType: "AUTO" }
  };
}

function ramp(value) {
  if (value < -0.1) return [0.14, 0.18, 0.32];
  if (value < 0.2) return [0.86, 0.78, 0.55];
  if (value < 0.45) return [0.48, 0.68, 0.33];
  return [0.12, 0.44, 0.24];
}

function evaluatePixel(sample) {
  let ndvi = index(sample.B08, sample.B04);
  let color = ramp(ndvi);
  return [color[0], color[1], color[2], sample.dataMask];
}
""".strip(),
    "ndmi": """
//VERSION=3
function setup() {
  return {
    input: ["B08", "B11", "dataMask"],
    output: { bands: 4, sampleType: "AUTO" }
  };
}

function ramp(value) {
  if (value < -0.2) return [0.48, 0.28, 0.16];
  if (value < 0.0) return [0.80, 0.64, 0.38];
  if (value < 0.25) return [0.44, 0.68, 0.70];
  return [0.12, 0.34, 0.62];
}

function evaluatePixel(sample) {
  let ndmi = index(sample.B08, sample.B11);
  let color = ramp(ndmi);
  return [color[0], color[1], color[2], sample.dataMask];
}
""".strip(),
}


@dataclass(frozen=True)
class Period:
    year: int
    month: int

    @property
    def key(self) -> str:
        return f"{self.year}-{self.month:02d}"

    @property
    def start(self) -> str:
        return f"{self.year}-{self.month:02d}-01T00:00:00Z"

    @property
    def end(self) -> str:
        day = calendar.monthrange(self.year, self.month)[1]
        return f"{self.year}-{self.month:02d}-{day:02d}T23:59:59Z"


def json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ENV_TRUE_VALUES


def cdse_uses_env_proxy() -> bool:
    return env_flag("CDSE_USE_ENV_PROXY", default=False)


def configured_proxy_keys() -> list[str]:
    return [key for key in PROXY_ENV_KEYS if os.environ.get(key)]


def cdse_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = cdse_uses_env_proxy()
    return session


def cdse_post(url: str, **kwargs: Any) -> requests.Response:
    try:
        with cdse_session() as session:
            return session.post(url, **kwargs)
    except requests.exceptions.ProxyError as exc:
        proxy_keys = ", ".join(configured_proxy_keys()) or "unknown proxy env"
        raise RuntimeError(
            "CDSE request failed through an environment proxy. "
            f"Proxy variables seen: {proxy_keys}. "
            "Set CDSE_USE_ENV_PROXY=0 or remove broken HTTP(S)_PROXY values."
        ) from exc


def cloud_cover(item: dict[str, Any]) -> float:
    value = item.get("properties", {}).get("eo:cloud_cover")
    if value is None:
        return 10_000.0
    return float(value)


def stac_search_s2(
    bbox: tuple[float, float, float, float],
    period: Period,
    max_cloud: float,
    limit: int = 10,
    use_cache: bool = True,
) -> dict[str, Any]:
    max_cloud = float(max_cloud)
    payload = {
        "collections": ["sentinel-2-l2a"],
        "bbox": list(bbox),
        "datetime": f"{period.start}/{period.end}",
        "limit": limit,
        "query": {
            "eo:cloud_cover": {"lte": max_cloud},
        },
        "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        "fields": {
            "include": [
                "id",
                "collection",
                "bbox",
                "properties.datetime",
                "properties.eo:cloud_cover",
                "properties.s2:mgrs_tile",
                "properties.sat:orbit_state",
                "properties.platform",
            ],
            "exclude": ["assets"],
        },
    }
    cache_path = STAC_CACHE_DIR / f"s2_{period.key}_{json_hash(payload)}.json"
    if use_cache:
        cached = read_json(cache_path)
        if cached is not None:
            cached["cache_status"] = "hit"
            return cached

    response = cdse_post(STAC_SEARCH_URL, json=payload, timeout=60)
    if response.status_code >= 400:
        raise RuntimeError(
            f"STAC search failed: HTTP {response.status_code}: {response.text[:2000]}"
        )

    result = response.json()
    features = result.get("features", [])
    features.sort(key=cloud_cover)
    normalized = {
        "period": period.key,
        "request": payload,
        "matched": result.get("context", {}).get("matched"),
        "items": [normalize_stac_item(item) for item in features],
        "cache_status": "miss",
    }
    write_json(cache_path, normalized)
    return normalized


def normalize_stac_item(item: dict[str, Any]) -> dict[str, Any]:
    props = item.get("properties", {})
    return {
        "id": item.get("id"),
        "collection": item.get("collection"),
        "bbox": item.get("bbox"),
        "datetime": props.get("datetime"),
        "cloud_cover": props.get("eo:cloud_cover"),
        "mgrs_tile": props.get("s2:mgrs_tile"),
        "orbit_state": props.get("sat:orbit_state"),
        "platform": props.get("platform"),
    }


def best_s2_observations(
    bbox: tuple[float, float, float, float],
    years: list[int],
    months: list[int],
    max_cloud: float,
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for year in years:
        for month in months:
            period = Period(year=year, month=month)
            result = stac_search_s2(bbox=bbox, period=period, max_cloud=max_cloud)
            item = result["items"][0] if result["items"] else None
            observations.append(
                {
                    "period": period.key,
                    "year": year,
                    "month": month,
                    "matched": result.get("matched"),
                    "cache_status": result.get("cache_status"),
                    "best_item": item,
                    "items": result["items"],
                }
            )
    return observations


def get_access_token() -> str:
    client_id = (
        os.environ.get("CDSE_SH_CLIENT_ID")
        or os.environ.get("CDSE_CLIENT_ID")
        or os.environ.get("SH_CLIENT_ID")
    )
    client_secret = (
        os.environ.get("CDSE_SH_CLIENT_SECRET")
        or os.environ.get("CDSE_CLIENT_SECRET")
        or os.environ.get("SH_CLIENT_SECRET")
    )
    if not client_id or not client_secret:
        raise RuntimeError(
            "CDSE credentials are required. Put CDSE_SH_CLIENT_ID and "
            "CDSE_SH_CLIENT_SECRET into the project .env file."
        )

    response = cdse_post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=60,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"Could not fetch CDSE token: HTTP {response.status_code}: "
            f"{response.text[:1000]}"
        )
    return response.json()["access_token"]


def parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def tight_time_range(datetime_value: str) -> tuple[str, str]:
    center = parse_datetime(datetime_value)
    if center.tzinfo is None:
        center = center.replace(tzinfo=timezone.utc)
    start = center - timedelta(minutes=30)
    end = center + timedelta(minutes=30)
    return (
        start.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        end.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def bbox_to_web_mercator(
    bbox: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    west, south, east, north = bbox
    min_x, min_y = TO_WEB_MERCATOR.transform(west, south)
    max_x, max_y = TO_WEB_MERCATOR.transform(east, north)
    return min_x, min_y, max_x, max_y


def process_request(
    bbox: tuple[float, float, float, float],
    time_from: str,
    time_to: str,
    evalscript: str,
    width: int,
    height: int,
    output_type: str,
    max_cloud: float,
    crs: str = CRS84,
) -> dict[str, Any]:
    return {
        "input": {
            "bounds": {
                "bbox": list(bbox),
                "properties": {"crs": crs},
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {"from": time_from, "to": time_to},
                        "mosaickingOrder": "leastCC",
                        "maxCloudCoverage": max_cloud,
                    },
                    "processing": {
                        "upsampling": "BILINEAR",
                        "downsampling": "BILINEAR",
                    },
                }
            ],
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [
                {
                    "identifier": "default",
                    "format": {"type": output_type},
                }
            ],
        },
        "evalscript": evalscript,
    }


def run_process_api(payload: dict[str, Any], accept: str) -> tuple[bytes, str]:
    token = get_access_token()
    response = cdse_post(
        PROCESS_URL,
        json=payload,
        headers={"Authorization": f"Bearer {token}", "Accept": accept},
        timeout=120,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"Process API failed: HTTP {response.status_code}: {response.text[:2000]}"
        )
    return response.content, response.headers.get("Content-Type", "")


def preview_filename(
    item_id: str,
    view: str,
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> str:
    key = json_hash(
        {
            "item_id": item_id,
            "view": view,
            "bbox": bbox,
            "width": width,
            "height": height,
            "crs": WEB_MERCATOR_CRS,
            "version": 2,
        }
    )
    return f"{key}_{view}.png"


def source_filename(
    item_id: str,
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> str:
    key = json_hash({"item_id": item_id, "bbox": bbox, "width": width, "height": height})
    return f"{key}_s2_l2a_raw.tif"


def save_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_bytes(content)
    tmp_path.replace(path)


def fetch_s2_preview(
    bbox: tuple[float, float, float, float],
    item: dict[str, Any],
    view: str,
    width: int,
    height: int,
    max_cloud: float,
) -> dict[str, Any]:
    if view not in S2_VIEW_EVALSCRIPTS:
        raise ValueError(f"Unknown Sentinel-2 view: {view}")
    if not item.get("datetime") or not item.get("id"):
        raise ValueError("A STAC item with id and datetime is required.")

    out_path = PREVIEW_CACHE_DIR / preview_filename(item["id"], view, bbox, width, height)
    metadata_path = out_path.with_suffix(".json")
    if out_path.exists() and metadata_path.exists():
        metadata = read_json(metadata_path) or {}
        metadata["cache_status"] = "hit"
        return metadata

    time_from, time_to = tight_time_range(item["datetime"])
    request_bbox = bbox_to_web_mercator(bbox)
    payload = process_request(
        bbox=request_bbox,
        time_from=time_from,
        time_to=time_to,
        evalscript=S2_VIEW_EVALSCRIPTS[view],
        width=width,
        height=height,
        output_type="image/png",
        max_cloud=max_cloud,
        crs=WEB_MERCATOR_CRS,
    )
    content, content_type = run_process_api(payload, accept="image/png")
    save_bytes(out_path, content)
    metadata = {
        "view": view,
        "item": item,
        "bbox": bbox,
        "request_bbox": request_bbox,
        "request_crs": WEB_MERCATOR_CRS,
        "width": width,
        "height": height,
        "path": str(out_path),
        "url": f"/cache/previews/{out_path.name}",
        "content_type": content_type,
        "bytes": len(content),
        "request": payload,
        "cache_status": "miss",
    }
    write_json(metadata_path, metadata)
    return metadata


def fetch_s2_source(
    bbox: tuple[float, float, float, float],
    item: dict[str, Any],
    width: int,
    height: int,
    max_cloud: float,
) -> dict[str, Any]:
    if not item.get("datetime") or not item.get("id"):
        raise ValueError("A STAC item with id and datetime is required.")

    out_path = S2_CACHE_DIR / source_filename(item["id"], bbox, width, height)
    metadata_path = out_path.with_suffix(".json")
    if out_path.exists() and metadata_path.exists():
        metadata = read_json(metadata_path) or {}
        metadata["cache_status"] = "hit"
        return metadata

    time_from, time_to = tight_time_range(item["datetime"])
    payload = process_request(
        bbox=bbox,
        time_from=time_from,
        time_to=time_to,
        evalscript=S2_RAW_EVALSCRIPT,
        width=width,
        height=height,
        output_type="image/tiff",
        max_cloud=max_cloud,
    )
    content, content_type = run_process_api(payload, accept="image/tiff")
    save_bytes(out_path, content)
    metadata = {
        "item": item,
        "bbox": bbox,
        "width": width,
        "height": height,
        "bands": ["B02", "B03", "B04", "B08", "B11", "B12", "SCL", "dataMask"],
        "spatial_resolution_note": (
            "Output grid is requested at 10 m. B11/B12 native resolution is 20 m "
            "and is resampled for co-display; this does not add physical detail."
        ),
        "preprocessing": {
            "upsampling": "BILINEAR",
            "downsampling": "BILINEAR",
            "mosaickingOrder": "leastCC",
        },
        "path": str(out_path),
        "content_type": content_type,
        "bytes": len(content),
        "request": payload,
        "cache_status": "miss",
    }
    write_json(metadata_path, metadata)
    return metadata

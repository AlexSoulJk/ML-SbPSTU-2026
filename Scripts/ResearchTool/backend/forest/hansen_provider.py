from __future__ import annotations

from typing import Callable

import requests

from ..config import FOREST_HANSEN_CACHE_DIR
from .models import HansenTile


HANSEN_VERSION = "GFC-2025-v1.13"
HANSEN_BASE_URL = f"https://storage.googleapis.com/earthenginepartners-hansen/{HANSEN_VERSION}"
HANSEN_LAYERS = {"lossyear", "treecover2000"}
ProgressCallback = Callable[[str, str], None]
CancelCallback = Callable[[], bool]


def tile_id_for_lonlat(lon: float, lat: float) -> str:
    lat_top = int((lat + 9.999999) // 10 * 10)
    lon_left = int(lon // 10 * 10)

    lat_hemisphere = "N" if lat_top >= 0 else "S"
    lon_hemisphere = "E" if lon_left >= 0 else "W"
    return f"{abs(lat_top):02d}{lat_hemisphere}_{abs(lon_left):03d}{lon_hemisphere}"


def hansen_tile_url(layer: str, tile_id: str) -> str:
    if layer not in HANSEN_LAYERS:
        raise ValueError(f"Unsupported Hansen layer: {layer}")
    return f"{HANSEN_BASE_URL}/Hansen_{HANSEN_VERSION}_{layer}_{tile_id}.tif"


def hansen_tile_path(layer: str, tile_id: str) -> str:
    return str(FOREST_HANSEN_CACHE_DIR / HANSEN_VERSION / layer / f"{tile_id}.tif")


def format_mb(size_bytes: int) -> str:
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def download_hansen_tile(
    layer: str,
    tile_id: str,
    *,
    progress: ProgressCallback | None = None,
    should_cancel: CancelCallback | None = None,
) -> str:
    local_path = FOREST_HANSEN_CACHE_DIR / HANSEN_VERSION / layer / f"{tile_id}.tif"
    if local_path.exists():
        if progress:
            progress("cache", f"{layer}: cache hit for {tile_id}")
        return str(local_path)

    local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = local_path.with_suffix(".tif.tmp")
    session = requests.Session()
    session.trust_env = False
    if progress:
        progress("cache", f"{layer}: downloading Hansen tile {tile_id}")
    with session.get(hansen_tile_url(layer, tile_id), stream=True, timeout=120) as response:
        if response.status_code >= 400:
            raise RuntimeError(
                f"Hansen tile download failed: HTTP {response.status_code}: {response.text[:500]}"
            )
        total_bytes = int(response.headers.get("content-length") or 0)
        downloaded = 0
        last_reported = 0
        with tmp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    if should_cancel and should_cancel():
                        tmp_path.unlink(missing_ok=True)
                        raise RuntimeError("Hansen tile download cancelled.")
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if progress and (downloaded - last_reported >= 8 * 1024 * 1024):
                        last_reported = downloaded
                        if total_bytes:
                            progress(
                                "cache",
                                f"{layer}: downloading {format_mb(downloaded)} / {format_mb(total_bytes)}",
                            )
                        else:
                            progress("cache", f"{layer}: downloading {format_mb(downloaded)}")
    if should_cancel and should_cancel():
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError("Hansen tile download cancelled.")
    tmp_path.replace(local_path)
    if progress:
        progress("cache", f"{layer}: cached {tile_id}")
    return str(local_path)


def tile_model(layer: str, tile_id: str, local_path: str) -> HansenTile:
    return HansenTile(
        layer=layer,
        tile_id=tile_id,
        version=HANSEN_VERSION,
        local_path=local_path,
    )

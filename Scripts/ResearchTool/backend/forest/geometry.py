from __future__ import annotations

import math
from typing import Any

from pyproj import Transformer
from shapely.geometry import Point, Polygon, mapping
from shapely.ops import transform


DEFAULT_SAMPLE_PLOT_M = 1000.0
DEFAULT_VIEWER_AOI_M = 2000.0


def utm_epsg_for_lonlat(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def transformers_for_lonlat(lon: float, lat: float) -> tuple[Transformer, Transformer, int]:
    epsg = utm_epsg_for_lonlat(lon, lat)
    to_metric = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    to_lonlat = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    return to_metric, to_lonlat, epsg


def square_around_point(lon: float, lat: float, side_m: float) -> tuple[dict[str, Any], list[float]]:
    to_metric, to_lonlat, _epsg = transformers_for_lonlat(lon, lat)
    center_m = transform(to_metric.transform, Point(lon, lat))
    half = side_m / 2
    square_m = Polygon(
        [
            (center_m.x - half, center_m.y - half),
            (center_m.x + half, center_m.y - half),
            (center_m.x + half, center_m.y + half),
            (center_m.x - half, center_m.y + half),
            (center_m.x - half, center_m.y - half),
        ]
    )
    square_lonlat = transform(to_lonlat.transform, square_m)
    min_x, min_y, max_x, max_y = square_lonlat.bounds
    return mapping(square_lonlat), [min_x, min_y, max_x, max_y]


def sample_geometries(
    lon: float,
    lat: float,
    sample_plot_m: float = DEFAULT_SAMPLE_PLOT_M,
    viewer_aoi_m: float = DEFAULT_VIEWER_AOI_M,
) -> dict[str, Any]:
    plot, plot_bbox = square_around_point(lon, lat, sample_plot_m)
    viewer_aoi, viewer_bbox = square_around_point(lon, lat, viewer_aoi_m)
    return {
        "point": {"type": "Point", "coordinates": [lon, lat]},
        "sample_plot": plot,
        "sample_plot_bbox": plot_bbox,
        "sample_plot_size_m": sample_plot_m,
        "viewer_aoi": viewer_aoi,
        "viewer_aoi_bbox": viewer_bbox,
        "viewer_aoi_size_m": viewer_aoi_m,
    }


def haversine_m(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    radius_m = 6_371_000.0
    phi_a = math.radians(lat_a)
    phi_b = math.radians(lat_b)
    delta_phi = math.radians(lat_b - lat_a)
    delta_lambda = math.radians(lon_b - lon_a)
    hav = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi_a) * math.cos(phi_b) * math.sin(delta_lambda / 2) ** 2
    )
    return 2 * radius_m * math.atan2(math.sqrt(hav), math.sqrt(1 - hav))

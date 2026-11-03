from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def _package_dir(package_name: str) -> Path | None:
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        return None
    if spec.submodule_search_locations:
        return Path(next(iter(spec.submodule_search_locations))).resolve()
    if spec.origin:
        return Path(spec.origin).resolve().parent
    return None


def configure_geospatial_environment() -> dict[str, str]:
    """Use GDAL/PROJ data bundled with the active Rasterio installation.

    System-wide PostgreSQL/PostGIS installations commonly define PROJ_LIB and
    GDAL_DATA. Those files can be incompatible with Rasterio's bundled GDAL and
    make valid EPSG codes unavailable. Resolve the directories from the active
    Python environment instead of relying on machine-specific paths.
    """

    rasterio_dir = _package_dir("rasterio")
    if rasterio_dir is None:
        return {}

    configured: dict[str, str] = {}
    proj_dir = rasterio_dir / "proj_data"
    if (proj_dir / "proj.db").is_file():
        proj_value = str(proj_dir)
        os.environ["PROJ_DATA"] = proj_value
        os.environ["PROJ_LIB"] = proj_value
        configured["PROJ_DATA"] = proj_value
        configured["PROJ_LIB"] = proj_value

    gdal_dir = rasterio_dir / "gdal_data"
    if gdal_dir.is_dir():
        gdal_value = str(gdal_dir)
        os.environ["GDAL_DATA"] = gdal_value
        configured["GDAL_DATA"] = gdal_value

    return configured

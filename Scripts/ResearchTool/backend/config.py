from __future__ import annotations

from pathlib import Path


TOOL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TOOL_ROOT.parents[1]
FRONTEND_DIR = TOOL_ROOT / "frontend"
DATA_DIR = TOOL_ROOT / "Data"
SOURCE_DIR = DATA_DIR / "source"
CACHE_DIR = DATA_DIR / "cache"
STAC_CACHE_DIR = CACHE_DIR / "stac"
S2_CACHE_DIR = CACHE_DIR / "sentinel2"
PREVIEW_CACHE_DIR = CACHE_DIR / "previews"
PROCESSED_DIR = DATA_DIR / "processed"
FOREST_DATA_DIR = DATA_DIR / "forest"
FOREST_IMPORTS_DIR = FOREST_DATA_DIR / "imports"
FOREST_EXPORTS_DIR = FOREST_DATA_DIR / "exports"
FOREST_STATE_DB_PATH = FOREST_DATA_DIR / "state.sqlite"
FOREST_CACHE_DIR = CACHE_DIR / "forest"
FOREST_HANSEN_CACHE_DIR = FOREST_CACHE_DIR / "hansen"
FOREST_SENTINEL_METADATA_CACHE_DIR = FOREST_CACHE_DIR / "sentinel" / "metadata"
FOREST_SENTINEL_RASTER_CACHE_DIR = FOREST_CACHE_DIR / "sentinel" / "rasters"
FOREST_DERIVED_CACHE_DIR = FOREST_CACHE_DIR / "derived"

DEFAULT_KML_PATH = SOURCE_DIR / "16_00-6.3746_boundary.kml"
FALLBACK_KML_PATH = PROJECT_ROOT / "TestData" / "kml" / "16_00-6.3746_boundary.kml"
ENV_FILE = PROJECT_ROOT / ".env"

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
STAC_SEARCH_URL = "https://stac.dataspace.copernicus.eu/v1/search"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/process/v1"
CRS84 = "http://www.opengis.net/def/crs/OGC/1.3/CRS84"

DEFAULT_YEARS = [2023, 2024, 2025]
DEFAULT_MONTHS = [5, 7, 9]
DEFAULT_SEGMENT_LENGTH_M = 2000
DEFAULT_CONTEXT_M = 200
DEFAULT_MAX_CLOUD = 30
S2_RESOLUTION_M = 10


def ensure_directories() -> None:
    for path in (
        SOURCE_DIR,
        STAC_CACHE_DIR,
        S2_CACHE_DIR,
        PREVIEW_CACHE_DIR,
        PROCESSED_DIR,
        FOREST_IMPORTS_DIR,
        FOREST_EXPORTS_DIR,
        FOREST_HANSEN_CACHE_DIR,
        FOREST_SENTINEL_METADATA_CACHE_DIR,
        FOREST_SENTINEL_RASTER_CACHE_DIR,
        FOREST_DERIVED_CACHE_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def parse_dotenv_value(raw_value: str) -> str:
    value = raw_value.strip()
    if not value:
        return ""

    quote = value[0]
    if quote not in {"'", '"'}:
        for marker in (" #", "\t#"):
            comment_start = value.find(marker)
            if comment_start != -1:
                value = value[:comment_start]
        return value.strip()

    chars: list[str] = []
    escaped = False
    for char in value[1:]:
        if escaped:
            chars.append({"n": "\n", "r": "\r", "t": "\t"}.get(char, char))
            escaped = False
            continue
        if quote == '"' and char == "\\":
            escaped = True
            continue
        if char == quote:
            break
        chars.append(char)
    return "".join(chars)


def is_env_key(key: str) -> bool:
    if not key or key[0].isdigit():
        return False
    return all(char.isalnum() or char == "_" for char in key)


def load_dotenv(env_file: Path = ENV_FILE) -> None:
    import os

    if not env_file.exists():
        return

    for line in env_file.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped.removeprefix("export ").lstrip()
        if "=" not in stripped:
            continue

        key, raw_value = stripped.split("=", 1)
        key = key.strip()
        if is_env_key(key) and key not in os.environ:
            os.environ[key] = parse_dotenv_value(raw_value)

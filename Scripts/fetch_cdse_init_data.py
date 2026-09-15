"""Fetch small Sentinel-1 GRD and Sentinel-2 L2A chips from CDSE.

The script uses the Copernicus Data Space Sentinel Hub Process API and writes
small GeoTIFF patches into TestData/Init by default. It requests the same AOIs
for several time windows, which makes the resulting files convenient for quick
change-detection or multimodal ML experiments.

Credentials:
    Create a Sentinel Hub OAuth client in the CDSE dashboard, then put these
    variables into the project .env file or set them in your shell:
        CDSE_SH_CLIENT_ID
        CDSE_SH_CLIENT_SECRET

Examples:
    python Scripts/fetch_cdse_init_data.py --dry-run
    python Scripts/fetch_cdse_init_data.py
    python Scripts/fetch_cdse_init_data.py --area spb_polytech --window summer_2024
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
PROCESS_URL = "https://sh.dataspace.copernicus.eu/process/v1"
CATALOG_URL = "https://sh.dataspace.copernicus.eu/catalog/v1/search"
CRS84 = "http://www.opengis.net/def/crs/OGC/1.3/CRS84"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "TestData" / "Init"


@dataclass(frozen=True)
class Area:
    name: str
    bbox: tuple[float, float, float, float]
    description: str


@dataclass(frozen=True)
class TimeWindow:
    name: str
    start: str
    end: str


@dataclass(frozen=True)
class CollectionSpec:
    key: str
    collection: str
    filename_token: str
    bands: tuple[str, ...]
    evalscript: str
    data_filter_extra: dict[str, Any]
    processing: dict[str, Any] | None = None


AREAS = (
    Area(
        name="spb_polytech",
        bbox=(30.3520, 59.9980, 30.3920, 60.0250),
        description="Saint Petersburg, Polytechnical University area",
    ),
    Area(
        name="kronstadt",
        bbox=(29.7150, 59.9650, 29.8050, 60.0200),
        description="Saint Petersburg, Kronstadt coast and water",
    ),
)

TIME_WINDOWS = (
    TimeWindow(name="spring_2024", start="2024-05-15", end="2024-05-31"),
    TimeWindow(name="summer_2024", start="2024-08-15", end="2024-08-31"),
)

S2_L2A_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B02", "B03", "B04", "B08", "dataMask"],
      units: "DN"
    }],
    output: {
      id: "default",
      bands: 5,
      sampleType: "UINT16"
    }
  };
}

function evaluatePixel(sample) {
  return [sample.B02, sample.B03, sample.B04, sample.B08, sample.dataMask];
}
""".strip()

S1_GRD_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["VV", "VH", "dataMask"]
    }],
    output: {
      id: "default",
      bands: 3,
      sampleType: "FLOAT32"
    }
  };
}

function evaluatePixel(sample) {
  return [sample.VV, sample.VH, sample.dataMask];
}
""".strip()

COLLECTIONS = (
    CollectionSpec(
        key="s2",
        collection="sentinel-2-l2a",
        filename_token="s2_l2a_b02_b03_b04_b08_mask",
        bands=("B02", "B03", "B04", "B08", "dataMask"),
        evalscript=S2_L2A_EVALSCRIPT,
        data_filter_extra={
            "mosaickingOrder": "leastCC",
            "maxCloudCoverage": 80,
        },
    ),
    CollectionSpec(
        key="s1",
        collection="sentinel-1-grd",
        filename_token="s1_grd_vv_vh_mask",
        bands=("VV", "VH", "dataMask"),
        evalscript=S1_GRD_EVALSCRIPT,
        data_filter_extra={
            "mosaickingOrder": "mostRecent",
            "acquisitionMode": "IW",
            "polarization": "DV",
            "resolution": "HIGH",
        },
        processing={
            "orthorectify": "true",
            "backCoeff": "SIGMA0_ELLIPSOID",
        },
    ),
)


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


def load_dotenv(env_file: Path = DEFAULT_ENV_FILE) -> None:
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


def parse_args() -> argparse.Namespace:
    area_choices = [area.name for area in AREAS]
    window_choices = [window.name for window in TIME_WINDOWS]
    collection_choices = [spec.key for spec in COLLECTIONS]

    parser = argparse.ArgumentParser(
        description=(
            "Download small aligned CDSE Sentinel-1 GRD and Sentinel-2 L2A "
            "GeoTIFF chips into TestData/Init."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--client-id",
        default=os.environ.get("CDSE_SH_CLIENT_ID")
        or os.environ.get("CDSE_CLIENT_ID")
        or os.environ.get("SH_CLIENT_ID"),
        help="CDSE Sentinel Hub OAuth client id. Can also be set via env.",
    )
    parser.add_argument(
        "--client-secret",
        default=os.environ.get("CDSE_SH_CLIENT_SECRET")
        or os.environ.get("CDSE_CLIENT_SECRET")
        or os.environ.get("SH_CLIENT_SECRET"),
        help="CDSE Sentinel Hub OAuth client secret. Can also be set via env.",
    )
    parser.add_argument(
        "--area",
        action="append",
        choices=area_choices,
        help="Area to fetch. May be passed multiple times. Default: all.",
    )
    parser.add_argument(
        "--window",
        action="append",
        choices=window_choices,
        help="Time window to fetch. May be passed multiple times. Default: all.",
    )
    parser.add_argument(
        "--collection",
        action="append",
        choices=collection_choices,
        help="Collection key to fetch: s1 or s2. Default: both.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=256,
        help="Output raster width in pixels. Default: 256.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="Output raster height in pixels. Default: 256.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="HTTP timeout in seconds. Default: 120.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload files that already exist.",
    )
    parser.add_argument(
        "--no-catalog-check",
        dest="catalog_check",
        action="store_false",
        help="Skip the lightweight STAC Catalog availability check.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print and write the planned requests; no authentication/download.",
    )
    return parser.parse_args()


def import_requests() -> Any:
    try:
        import requests
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: requests. Install it with "
            "`python -m pip install requests`."
        ) from exc
    return requests


def iso_range(window: TimeWindow) -> dict[str, str]:
    return {
        "from": f"{window.start}T00:00:00Z",
        "to": f"{window.end}T23:59:59Z",
    }


def build_process_request(
    area: Area,
    window: TimeWindow,
    spec: CollectionSpec,
    width: int,
    height: int,
) -> dict[str, Any]:
    data_filter = {
        "timeRange": iso_range(window),
        **spec.data_filter_extra,
    }
    data_item: dict[str, Any] = {
        "type": spec.collection,
        "dataFilter": data_filter,
    }
    if spec.processing:
        data_item["processing"] = spec.processing

    return {
        "input": {
            "bounds": {
                "bbox": list(area.bbox),
                "properties": {"crs": CRS84},
            },
            "data": [data_item],
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [
                {
                    "identifier": "default",
                    "format": {"type": "image/tiff"},
                }
            ],
        },
        "evalscript": spec.evalscript,
    }


def build_catalog_request(
    area: Area,
    window: TimeWindow,
    spec: CollectionSpec,
) -> dict[str, Any]:
    return {
        "bbox": list(area.bbox),
        "datetime": f"{window.start}T00:00:00Z/{window.end}T23:59:59Z",
        "collections": [spec.collection],
        "limit": 5,
        "fields": {
            "include": [
                "id",
                "properties.datetime",
                "properties.eo:cloud_cover",
                "properties.s1:polarization",
                "properties.sar:instrument_mode",
            ],
            "exclude": [],
        },
    }


def fetch_token(requests: Any, client_id: str, client_secret: str, timeout: int) -> str:
    response = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=timeout,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            "Could not fetch CDSE access token. "
            f"HTTP {response.status_code}: {response.text[:1000]}"
        )

    payload = response.json()
    try:
        return payload["access_token"]
    except KeyError as exc:
        raise RuntimeError(f"Token response has no access_token: {payload}") from exc


def catalog_summary(
    requests: Any,
    token: str,
    catalog_request: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    response = requests.post(
        CATALOG_URL,
        json=catalog_request,
        headers={"Authorization": f"Bearer {token}"},
        timeout=timeout,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            "Catalog request failed. "
            f"HTTP {response.status_code}: {response.text[:1000]}"
        )

    payload = response.json()
    features = payload.get("features", [])
    return {
        "returned": len(features),
        "matched": payload.get("context", {}).get("matched"),
        "sample_items": [
            {
                "id": item.get("id"),
                "datetime": item.get("properties", {}).get("datetime"),
                "cloud_cover": item.get("properties", {}).get("eo:cloud_cover"),
                "polarization": item.get("properties", {}).get("s1:polarization"),
                "instrument_mode": item.get("properties", {}).get(
                    "sar:instrument_mode"
                ),
            }
            for item in features
        ],
    }


def download_chip(
    requests: Any,
    token: str,
    process_request: dict[str, Any],
    out_path: Path,
    timeout: int,
) -> tuple[int, str]:
    response = requests.post(
        PROCESS_URL,
        json=process_request,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "image/tiff",
        },
        timeout=timeout,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            "Process request failed. "
            f"HTTP {response.status_code}: {response.text[:2000]}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_bytes(response.content)
    tmp_path.replace(out_path)

    return len(response.content), response.headers.get("Content-Type", "")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def select_by_name(items: tuple[Any, ...], selected_names: list[str] | None) -> list[Any]:
    if not selected_names:
        return list(items)
    selected = set(selected_names)
    return [
        item
        for item in items
        if getattr(item, "name", None) in selected or getattr(item, "key", None) in selected
    ]


def output_filename(area: Area, window: TimeWindow, spec: CollectionSpec) -> str:
    return f"{area.name}_{window.name}_{spec.filename_token}.tif"


def validate_args(args: argparse.Namespace) -> None:
    if args.width <= 0 or args.height <= 0:
        raise SystemExit("--width and --height must be positive integers.")

    for window in TIME_WINDOWS:
        datetime.strptime(window.start, "%Y-%m-%d")
        datetime.strptime(window.end, "%Y-%m-%d")
        if window.start > window.end:
            raise SystemExit(f"Invalid time window: {window.name}")

    if not args.dry_run and (not args.client_id or not args.client_secret):
        raise SystemExit(
            "CDSE credentials are required for download. Set "
            "CDSE_SH_CLIENT_ID/CDSE_SH_CLIENT_SECRET or pass "
            "--client-id/--client-secret. Use --dry-run to inspect the plan."
        )


def main() -> int:
    load_dotenv()
    args = parse_args()
    validate_args(args)

    areas = select_by_name(AREAS, args.area)
    windows = select_by_name(TIME_WINDOWS, args.window)
    specs = select_by_name(COLLECTIONS, args.collection)

    manifest: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "token_url": TOKEN_URL,
            "process_url": PROCESS_URL,
            "catalog_url": CATALOG_URL,
        },
        "output_dir": str(args.output_dir),
        "width": args.width,
        "height": args.height,
        "bbox_crs": CRS84,
        "areas": [asdict(area) for area in areas],
        "time_windows": [asdict(window) for window in windows],
        "items": [],
    }

    work_items = []
    for area in areas:
        for window in windows:
            for spec in specs:
                out_path = args.output_dir / output_filename(area, window, spec)
                process_request = build_process_request(
                    area=area,
                    window=window,
                    spec=spec,
                    width=args.width,
                    height=args.height,
                )
                catalog_request = build_catalog_request(area, window, spec)
                work_items.append((area, window, spec, out_path, process_request, catalog_request))

    if args.dry_run:
        for area, window, spec, out_path, process_request, catalog_request in work_items:
            manifest["items"].append(
                {
                    "area": area.name,
                    "time_window": window.name,
                    "collection": spec.collection,
                    "bands": spec.bands,
                    "path": str(out_path),
                    "status": "planned",
                    "process_request": process_request,
                    "catalog_request": catalog_request,
                }
            )
            print(f"PLAN {spec.collection:16s} {area.name:13s} {window.name:11s} -> {out_path}")
        write_json(args.output_dir / "manifest.plan.json", manifest)
        print(f"Wrote plan: {args.output_dir / 'manifest.plan.json'}")
        return 0

    requests = import_requests()
    token = fetch_token(
        requests=requests,
        client_id=args.client_id,
        client_secret=args.client_secret,
        timeout=args.timeout,
    )

    for area, window, spec, out_path, process_request, catalog_request in work_items:
        item: dict[str, Any] = {
            "area": area.name,
            "time_window": window.name,
            "collection": spec.collection,
            "bands": spec.bands,
            "path": str(out_path),
            "process_request": process_request,
            "catalog_request": catalog_request,
        }

        if out_path.exists() and not args.overwrite:
            item["status"] = "skipped_existing"
            item["bytes"] = out_path.stat().st_size
            manifest["items"].append(item)
            print(f"SKIP {out_path}")
            continue

        try:
            if args.catalog_check:
                item["catalog"] = catalog_summary(
                    requests=requests,
                    token=token,
                    catalog_request=catalog_request,
                    timeout=args.timeout,
                )
                if item["catalog"]["returned"] == 0:
                    item["status"] = "skipped_no_catalog_items"
                    manifest["items"].append(item)
                    print(
                        f"NO DATA {spec.collection:16s} "
                        f"{area.name:13s} {window.name:11s}"
                    )
                    continue

            started = time.monotonic()
            size_bytes, content_type = download_chip(
                requests=requests,
                token=token,
                process_request=process_request,
                out_path=out_path,
                timeout=args.timeout,
            )
            item.update(
                {
                    "status": "downloaded",
                    "bytes": size_bytes,
                    "content_type": content_type,
                    "elapsed_seconds": round(time.monotonic() - started, 2),
                }
            )
            print(f"OK   {size_bytes:9d} bytes -> {out_path}")
        except Exception as exc:
            item["status"] = "failed"
            item["error"] = str(exc)
            manifest["items"].append(item)
            write_json(args.output_dir / "manifest.json", manifest)
            raise

        manifest["items"].append(item)
        write_json(args.output_dir / "manifest.json", manifest)

    write_json(args.output_dir / "manifest.json", manifest)
    print(f"Wrote manifest: {args.output_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

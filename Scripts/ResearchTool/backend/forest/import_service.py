from __future__ import annotations

import csv
import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from ..config import FOREST_IMPORTS_DIR
from .models import ImportBatch, Sample


@dataclass(frozen=True)
class NormalizedSample:
    sample_id: str
    source_file: str
    source_format: str
    source_id: str | None
    lat: float
    lon: float
    source_event_year: int | None
    driver_primary: str | None
    driver_primary_code: str | None
    confidence_primary: str | None
    driver_secondary: str | None
    confidence_secondary: str | None
    region: str | None
    region_code: str | None
    tags: str | None
    notes: str | None
    raw_properties_json: str


def sanitize_filename(filename: str) -> str:
    safe = Path(filename).name.strip() or "upload.dat"
    return re.sub(r"[^A-Za-z0-9А-Яа-я._-]+", "_", safe)


def detect_format(filename: str, content: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in {".geojson", ".json"}:
        return "geojson"
    stripped = content.lstrip()
    if stripped.startswith(b"{"):
        return "geojson"
    return "csv"


def save_import_source(filename: str, content: bytes) -> Path:
    FOREST_IMPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stored_name = f"{uuid.uuid4().hex[:12]}_{sanitize_filename(filename)}"
    stored_path = FOREST_IMPORTS_DIR / stored_name
    stored_path.write_bytes(content)
    return stored_path


def normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", key.lower())


def lookup(properties: dict[str, Any], *candidates: str) -> Any:
    by_normalized = {normalize_key(str(key)): key for key in properties}
    for candidate in candidates:
        key = by_normalized.get(normalize_key(candidate))
        if key is not None:
            return properties.get(key)
    return None


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return str(value)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def parse_float(value: Any) -> float | None:
    text = clean_text(value)
    if text is None:
        return None
    try:
        return float(text.replace(",", "."))
    except ValueError:
        return None


def parse_int(value: Any) -> int | None:
    text = clean_text(value)
    if text is None:
        return None
    try:
        return int(float(text.replace(",", ".")))
    except ValueError:
        return None


def parse_source_event_year(properties: dict[str, Any]) -> int | None:
    direct_year = parse_int(
        lookup(
            properties,
            "source_event_year",
            "event_year",
            "event year",
            "loss_year",
            "loss year",
            "lossyear",
            "year",
        )
    )
    if direct_year is not None and 1980 <= direct_year <= 2100:
        return direct_year
    return None


def geometry_point(geometry: dict[str, Any] | None) -> tuple[float, float] | None:
    if not geometry or geometry.get("type") != "Point":
        return None
    coordinates = geometry.get("coordinates")
    if not isinstance(coordinates, list | tuple) or len(coordinates) < 2:
        return None
    lon = parse_float(coordinates[0])
    lat = parse_float(coordinates[1])
    if lat is None or lon is None:
        return None
    return lat, lon


def stable_sample_id(source_file: str, source_id: str | None, lat: float, lon: float) -> str:
    payload = json.dumps(
        {
            "source_file": source_file,
            "source_id": source_id,
            "lat": round(lat, 8),
            "lon": round(lon, 8),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def normalize_sample(
    properties: dict[str, Any],
    *,
    source_format: str,
    uploaded_filename: str,
    geometry: dict[str, Any] | None = None,
    fallback_source_id: str | None = None,
) -> NormalizedSample:
    point = geometry_point(geometry)
    lat = point[0] if point else parse_float(lookup(properties, "latitude", "lat", "y"))
    lon = point[1] if point else parse_float(lookup(properties, "longitude", "lon", "lng", "x"))
    if lat is None or lon is None:
        raise ValueError("No point geometry or lat/lon fields.")

    source_file = clean_text(lookup(properties, "source_file", "source file")) or uploaded_filename
    source_format_value = clean_text(lookup(properties, "source_format", "source format"))
    source_id = clean_text(lookup(properties, "id", "source_id", "source id")) or fallback_source_id
    raw_json = json.dumps(properties, ensure_ascii=False, sort_keys=True, default=str)

    return NormalizedSample(
        sample_id=stable_sample_id(source_file, source_id, lat, lon),
        source_file=source_file,
        source_format=source_format_value or source_format,
        source_id=source_id,
        lat=lat,
        lon=lon,
        source_event_year=parse_source_event_year(properties),
        driver_primary=clean_text(lookup(properties, "driver_primary", "driver primary", "driver")),
        driver_primary_code=clean_text(
            lookup(properties, "driver_primary_code", "driver primary code", "driver_code")
        ),
        confidence_primary=clean_text(
            lookup(properties, "confidence_primary", "confidence primary", "confidence")
        ),
        driver_secondary=clean_text(lookup(properties, "driver_secondary", "driver secondary")),
        confidence_secondary=clean_text(
            lookup(properties, "confidence_secondary", "confidence secondary")
        ),
        region=clean_text(lookup(properties, "region")),
        region_code=clean_text(lookup(properties, "region_code", "region code")),
        tags=clean_text(lookup(properties, "tags")),
        notes=clean_text(lookup(properties, "notes")),
        raw_properties_json=raw_json,
    )


def samples_from_csv(content: bytes, filename: str) -> tuple[list[NormalizedSample], list[dict[str, Any]]]:
    text = content.decode("utf-8-sig")
    reader = csv.DictReader(StringIO(text))
    samples: list[NormalizedSample] = []
    rejected: list[dict[str, Any]] = []
    for row_index, row in enumerate(reader, start=1):
        try:
            samples.append(
                normalize_sample(
                    dict(row),
                    source_format="csv",
                    uploaded_filename=filename,
                    fallback_source_id=str(row_index),
                )
            )
        except Exception as exc:
            rejected.append({"row": row_index, "reason": str(exc)})
    return samples, rejected


def samples_from_geojson(
    content: bytes,
    filename: str,
) -> tuple[list[NormalizedSample], list[dict[str, Any]]]:
    payload = json.loads(content.decode("utf-8-sig"))
    if payload.get("type") == "FeatureCollection":
        features = payload.get("features") or []
    elif payload.get("type") == "Feature":
        features = [payload]
    else:
        raise ValueError("GeoJSON must be a Feature or FeatureCollection.")

    samples: list[NormalizedSample] = []
    rejected: list[dict[str, Any]] = []
    for index, feature in enumerate(features, start=1):
        properties = feature.get("properties") or {}
        if not isinstance(properties, dict):
            properties = {}
        try:
            samples.append(
                normalize_sample(
                    properties,
                    source_format="geojson",
                    uploaded_filename=filename,
                    geometry=feature.get("geometry"),
                    fallback_source_id=str(index),
                )
            )
        except Exception as exc:
            rejected.append({"feature": index, "reason": str(exc)})
    return samples, rejected


def sample_to_model(sample: NormalizedSample) -> Sample:
    return Sample(
        sample_id=sample.sample_id,
        source_file=sample.source_file,
        source_format=sample.source_format,
        source_id=sample.source_id,
        lat=sample.lat,
        lon=sample.lon,
        source_event_year=sample.source_event_year,
        driver_primary=sample.driver_primary,
        driver_primary_code=sample.driver_primary_code,
        confidence_primary=sample.confidence_primary,
        driver_secondary=sample.driver_secondary,
        confidence_secondary=sample.confidence_secondary,
        region=sample.region,
        region_code=sample.region_code,
        tags=sample.tags,
        notes=sample.notes,
        raw_properties_json=sample.raw_properties_json,
    )


def import_file(
    db: Session,
    *,
    filename: str,
    content: bytes,
) -> dict[str, Any]:
    source_format = detect_format(filename, content)
    stored_path = save_import_source(filename, content)
    import_id = uuid.uuid4().hex

    imported_count = 0
    duplicate_count = 0
    rejected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    try:
        if source_format == "geojson":
            normalized, parse_rejected = samples_from_geojson(content, filename)
        else:
            normalized, parse_rejected = samples_from_csv(content, filename)
        rejected.extend(parse_rejected)
    except Exception as exc:
        rejected.append({"row": None, "reason": str(exc)})
        normalized = []

    for index, sample in enumerate(normalized, start=1):
        if sample.sample_id in seen_ids or db.get(Sample, sample.sample_id) is not None:
            duplicate_count += 1
            continue
        try:
            db.add(sample_to_model(sample))
            seen_ids.add(sample.sample_id)
            imported_count += 1
        except Exception as exc:
            rejected.append({"row": index, "reason": str(exc)})

    batch = ImportBatch(
        import_id=import_id,
        original_filename=filename,
        stored_path=str(stored_path),
        source_format=source_format,
        imported_count=imported_count,
        duplicate_count=duplicate_count,
        rejected_count=len(rejected),
        errors_json=json.dumps(rejected[:25], ensure_ascii=False),
    )
    db.add(batch)
    db.commit()

    return {
        "import_id": import_id,
        "filename": filename,
        "stored_path": str(stored_path),
        "source_format": source_format,
        "imported_count": imported_count,
        "duplicate_count": duplicate_count,
        "rejected_count": len(rejected),
        "rejected_examples": rejected[:10],
    }

from __future__ import annotations

import csv
import json
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shapely.geometry import MultiPolygon, Point, Polygon, box, mapping, shape
from shapely.ops import unary_union


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "TestData" / "Forest" / "Init"
OUTPUT_DIR = PROJECT_ROOT / "TestData" / "Forest" / "Init_Extracted"

CSV_GLOBS = ("*.csv",)
GEOJSON_GLOBS = ("*.geojson", "*.json")

LATITUDE_FIELD = "Latitude"
LONGITUDE_FIELD = "Longitude"
OUTPUT_SUFFIX = "_russia"
WRITE_COMBINED_CSV = True
WRITE_SUMMARY_JSON = True

# Optional precise mode:
# Put a countries GeoJSON path here, for example a Natural Earth admin_0 file.
# The script will select features whose name/admin fields match Russia.
RUSSIA_BOUNDARY_GEOJSON: Path | None = None
RUSSIA_NAME_VALUES = {
    "russia",
    "russian federation",
    "ru",
    "rus",
    "россия",
    "российская федерация",
}
RUSSIA_NAME_FIELDS = (
    "ADMIN",
    "Admin",
    "admin",
    "NAME",
    "Name",
    "name",
    "NAME_EN",
    "SOVEREIGNT",
    "SOVEREIGNT_EN",
    "ISO_A2",
    "ISO_A3",
)


def built_in_russia_geometry() -> Polygon | MultiPolygon:
    """Coarse built-in Russia geometry for point extraction.

    It is good enough for this dataset exploration, but for strict borders set
    RUSSIA_BOUNDARY_GEOJSON above to an external country boundary GeoJSON.
    """
    mainland = Polygon(
        [
            (27.2, 68.5),
            (30.0, 70.0),
            (42.0, 69.6),
            (53.0, 70.8),
            (67.0, 70.2),
            (82.0, 72.5),
            (97.0, 73.6),
            (118.0, 74.2),
            (139.0, 73.2),
            (160.0, 69.4),
            (180.0, 66.0),
            (180.0, 61.0),
            (170.0, 60.0),
            (164.0, 55.0),
            (158.0, 51.0),
            (149.0, 46.5),
            (140.5, 44.4),
            (136.0, 47.5),
            (132.0, 42.2),
            (128.0, 43.5),
            (124.0, 48.5),
            (118.0, 49.4),
            (112.0, 49.0),
            (106.0, 50.2),
            (100.0, 50.0),
            (94.0, 49.0),
            (88.0, 49.7),
            (83.0, 50.7),
            (77.0, 53.0),
            (70.0, 51.2),
            (62.0, 50.4),
            (55.0, 51.0),
            (51.0, 50.4),
            (48.2, 46.5),
            (43.5, 45.5),
            (39.8, 43.3),
            (37.0, 44.5),
            (38.5, 47.5),
            (40.0, 49.2),
            (36.0, 50.5),
            (32.0, 52.0),
            (31.0, 55.0),
            (30.2, 58.0),
            (28.4, 60.0),
            (31.0, 62.0),
            (31.0, 66.2),
            (27.2, 68.5),
        ]
    )

    extras = [
        box(19.4, 54.1, 22.95, 55.35),  # Kaliningrad oblast.
        box(145.0, 43.0, 156.8, 51.4),  # Kuril islands, coarse.
        box(-180.0, 63.0, -168.0, 72.5),  # Chukotka across antimeridian.
        box(35.0, 70.0, 180.0, 82.5),  # Arctic islands, coarse.
    ]
    return unary_union([mainland, *extras]).buffer(0)


def normalized_text(value: Any) -> str:
    return str(value).strip().casefold()


def feature_matches_russia(properties: dict[str, Any]) -> bool:
    for field in RUSSIA_NAME_FIELDS:
        value = properties.get(field)
        if value is not None and normalized_text(value) in RUSSIA_NAME_VALUES:
            return True
    return False


def load_russia_geometry() -> tuple[Polygon | MultiPolygon, str]:
    if RUSSIA_BOUNDARY_GEOJSON:
        path = RUSSIA_BOUNDARY_GEOJSON.expanduser().resolve()
        data = json.loads(path.read_text(encoding="utf-8"))
        geometries = []

        if data.get("type") == "FeatureCollection":
            for feature in data.get("features", []):
                properties = feature.get("properties") or {}
                geometry = feature.get("geometry")
                if geometry and feature_matches_russia(properties):
                    geometries.append(shape(geometry))
        elif data.get("type") == "Feature":
            geometries.append(shape(data["geometry"]))
        elif data.get("type") in {"Polygon", "MultiPolygon"}:
            geometries.append(shape(data))

        if not geometries:
            raise RuntimeError(f"No Russia boundary feature found in {path}")
        return unary_union(geometries).buffer(0), str(path)

    return built_in_russia_geometry(), "built_in_approx_russia_geometry"


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def point_from_record(record: dict[str, Any]) -> Point | None:
    lat = parse_float(record.get(LATITUDE_FIELD))
    lon = parse_float(record.get(LONGITUDE_FIELD))
    if lat is None or lon is None:
        return None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return Point(lon, lat)


def point_from_feature(feature: dict[str, Any]) -> Point | None:
    properties = feature.get("properties") or {}
    point = point_from_record(properties)
    if point is not None:
        return point

    geometry = feature.get("geometry") or {}
    if geometry.get("type") == "Point":
        coordinates = geometry.get("coordinates") or []
        if len(coordinates) >= 2:
            lon = parse_float(coordinates[0])
            lat = parse_float(coordinates[1])
            if lat is not None and lon is not None:
                return Point(lon, lat)
    return None


def is_russia_point(point: Point | None, russia_geometry: Polygon | MultiPolygon) -> bool:
    return point is not None and russia_geometry.covers(point)


def output_path_for(input_path: Path, suffix: str = OUTPUT_SUFFIX) -> Path:
    return OUTPUT_DIR / f"{input_path.stem}{suffix}{input_path.suffix}"


def add_filter_columns(record: dict[str, Any], geometry_source: str) -> dict[str, Any]:
    result = dict(record)
    result["Country_Filter"] = "Russia"
    result["Country_Filter_Source"] = geometry_source
    return result


def write_csv(path: Path, rows: list[dict[str, Any]], preferred_fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(preferred_fields)
    seen = set(fieldnames)
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def extract_csv(
    path: Path,
    russia_geometry: Polygon | MultiPolygon,
    geometry_source: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        source_fields = reader.fieldnames or []

    extracted = [
        add_filter_columns(row, geometry_source)
        for row in rows
        if is_russia_point(point_from_record(row), russia_geometry)
    ]
    output_path = output_path_for(path)
    write_csv(output_path, extracted, [*source_fields, "Country_Filter", "Country_Filter_Source"])

    combined = [
        {"Source_File": path.name, "Source_Format": "csv", **row}
        for row in extracted
    ]
    return (
        {
            "input_file": str(path),
            "output_file": str(output_path),
            "format": "csv",
            "input_rows": len(rows),
            "extracted_rows": len(extracted),
            "drivers": dict(Counter(row.get("Driver_primary", "") for row in extracted)),
        },
        combined,
    )


def extract_geojson(
    path: Path,
    russia_geometry: Polygon | MultiPolygon,
    geometry_source: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    features = data.get("features", [])
    extracted_features = []
    combined = []

    for feature in features:
        if not is_russia_point(point_from_feature(feature), russia_geometry):
            continue
        copied = deepcopy(feature)
        copied.setdefault("properties", {})
        copied["properties"] = add_filter_columns(copied["properties"], geometry_source)
        extracted_features.append(copied)
        combined.append(
            {
                "Source_File": path.name,
                "Source_Format": "geojson",
                **copied["properties"],
            }
        )

    output = {
        "type": "FeatureCollection",
        "name": f"{data.get('name', path.stem)}{OUTPUT_SUFFIX}",
        "features": extracted_features,
    }
    output_path = output_path_for(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return (
        {
            "input_file": str(path),
            "output_file": str(output_path),
            "format": "geojson",
            "input_features": len(features),
            "extracted_features": len(extracted_features),
            "drivers": dict(
                Counter(feature["properties"].get("Driver_primary", "") for feature in extracted_features)
            ),
        },
        combined,
    )


def iter_input_files(globs: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for pattern in globs:
        files.extend(INPUT_DIR.glob(pattern))
    return sorted(set(files))


def write_combined_csv(rows: list[dict[str, Any]]) -> str | None:
    if not WRITE_COMBINED_CSV:
        return None

    path = OUTPUT_DIR / "russia_points_combined.csv"
    preferred = [
        "Source_File",
        "Source_Format",
        "ID",
        "Latitude",
        "Longitude",
        "Driver_primary_code",
        "Driver_primary",
        "Confidence_primary",
        "Driver_secondary",
        "Confidence_secondary",
        "Region",
        "Region_code",
        "Tags",
        "Notes",
        "Country_Filter",
        "Country_Filter_Source",
    ]
    write_csv(path, rows, preferred)
    return str(path)


def write_summary(summary: dict[str, Any]) -> str | None:
    if not WRITE_SUMMARY_JSON:
        return None

    path = OUTPUT_DIR / "extraction_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    russia_geometry, geometry_source = load_russia_geometry()

    file_summaries = []
    combined_rows: list[dict[str, Any]] = []

    for path in iter_input_files(CSV_GLOBS):
        file_summary, rows = extract_csv(path, russia_geometry, geometry_source)
        file_summaries.append(file_summary)
        combined_rows.extend(rows)

    for path in iter_input_files(GEOJSON_GLOBS):
        file_summary, rows = extract_geojson(path, russia_geometry, geometry_source)
        file_summaries.append(file_summary)
        combined_rows.extend(rows)

    combined_csv = write_combined_csv(combined_rows)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(INPUT_DIR),
        "output_dir": str(OUTPUT_DIR),
        "geometry_source": geometry_source,
        "note": (
            "Russia is selected by point-in-polygon. The built-in geometry is coarse; "
            "set RUSSIA_BOUNDARY_GEOJSON for strict country borders."
        ),
        "files": file_summaries,
        "combined_csv": combined_csv,
        "total_extracted": len(combined_rows),
        "combined_driver_counts": dict(Counter(row.get("Driver_primary", "") for row in combined_rows)),
    }
    summary_path = write_summary(summary)

    print(f"Input: {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Geometry: {geometry_source}")
    for item in file_summaries:
        input_count = item.get("input_rows", item.get("input_features"))
        output_count = item.get("extracted_rows", item.get("extracted_features"))
        print(f"{Path(item['input_file']).name}: {output_count}/{input_count}")
    if combined_csv:
        print(f"Combined CSV: {combined_csv}")
    if summary_path:
        print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any


KML_NS = {"k": "http://www.opengis.net/kml/2.2"}
GEOMETRY_TAGS = ("Point", "LineString", "LinearRing", "Polygon", "MultiGeometry")


@dataclass(frozen=True)
class KmlGeometry:
    id: str
    name: str
    geometry_type: str
    coordinates: list[tuple[float, float]]
    style_url: str | None = None
    attributes: dict[str, str] | None = None


@dataclass(frozen=True)
class KmlDocument:
    name: str
    attributes: dict[str, str]
    geometries: list[KmlGeometry]
    bbox: tuple[float, float, float, float] | None


def safe_id(value: str, fallback: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip()).strip("_")
    return value.lower() or fallback


def parse_extended_data(node: ET.Element | None) -> dict[str, str]:
    if node is None:
        return {}
    attrs: dict[str, str] = {}
    for data in node.findall(".//k:Data", KML_NS):
        key = data.attrib.get("name")
        value_node = data.find("k:value", KML_NS)
        if key and value_node is not None and value_node.text is not None:
            attrs[key] = value_node.text.strip()
    return attrs


def parse_coordinate_text(text: str | None) -> list[tuple[float, float]]:
    if not text:
        return []

    coordinates: list[tuple[float, float]] = []
    for token in text.split():
        parts = token.split(",")
        if len(parts) < 2:
            continue
        coordinates.append((float(parts[0]), float(parts[1])))
    return coordinates


def find_geometry_type(placemark: ET.Element) -> str:
    types = [
        tag
        for tag in GEOMETRY_TAGS
        if placemark.find(f".//k:{tag}", KML_NS) is not None
    ]
    return "+".join(types) if types else "Unknown"


def collect_coordinates(placemark: ET.Element) -> list[tuple[float, float]]:
    coordinates: list[tuple[float, float]] = []
    for node in placemark.findall(".//k:coordinates", KML_NS):
        coordinates.extend(parse_coordinate_text(node.text))
    return coordinates


def compute_bbox(
    geometries: list[KmlGeometry],
) -> tuple[float, float, float, float] | None:
    points = [coord for geometry in geometries for coord in geometry.coordinates]
    if not points:
        return None
    lons = [lon for lon, _lat in points]
    lats = [lat for _lon, lat in points]
    return min(lons), min(lats), max(lons), max(lats)


def parse_kml_text(text: str) -> KmlDocument:
    root = ET.fromstring(text)
    document = root.find(".//k:Document", KML_NS)
    source = document if document is not None else root

    name_node = source.find("k:name", KML_NS)
    name = name_node.text.strip() if name_node is not None and name_node.text else ""
    attributes = parse_extended_data(source.find("k:ExtendedData", KML_NS))

    geometries: list[KmlGeometry] = []
    for index, placemark in enumerate(root.findall(".//k:Placemark", KML_NS), start=1):
        placemark_name_node = placemark.find("k:name", KML_NS)
        placemark_name = (
            placemark_name_node.text.strip()
            if placemark_name_node is not None and placemark_name_node.text
            else f"Placemark {index}"
        )
        style_node = placemark.find("k:styleUrl", KML_NS)
        style_url = style_node.text.strip() if style_node is not None and style_node.text else None

        coordinates = collect_coordinates(placemark)
        geometries.append(
            KmlGeometry(
                id=safe_id(placemark_name, f"placemark_{index}"),
                name=placemark_name,
                geometry_type=find_geometry_type(placemark),
                coordinates=coordinates,
                style_url=style_url,
                attributes=parse_extended_data(
                    placemark.find("k:ExtendedData", KML_NS)
                ),
            )
        )

    return KmlDocument(
        name=name,
        attributes=attributes,
        geometries=geometries,
        bbox=compute_bbox(geometries),
    )


def parse_kml_file(path: Path) -> KmlDocument:
    return parse_kml_text(path.read_text(encoding="utf-8-sig"))


def is_closed_line(coordinates: list[tuple[float, float]]) -> bool:
    if len(coordinates) < 4:
        return False
    return coordinates[0] == coordinates[-1]


def kml_document_to_api(document: KmlDocument) -> dict[str, Any]:
    return {
        "name": document.name,
        "attributes": document.attributes,
        "bbox": document.bbox,
        "geometries": [
            {
                "id": geometry.id,
                "name": geometry.name,
                "geometry_type": geometry.geometry_type,
                "coordinates": geometry.coordinates,
                "closed": is_closed_line(geometry.coordinates),
                "style_url": geometry.style_url,
                "attributes": geometry.attributes or {},
            }
            for geometry in document.geometries
        ],
    }

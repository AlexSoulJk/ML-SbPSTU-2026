from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from pyproj import Transformer
from shapely.geometry import LineString, Point, Polygon, mapping
from shapely.ops import transform
from shapely.validation import make_valid
from skimage.draw import polygon as draw_polygon
from skimage.morphology import medial_axis


@dataclass(frozen=True)
class Segment:
    id: str
    index: int
    start_m: float
    end_m: float
    length_m: float
    line_lonlat: LineString
    aoi_lonlat: Polygon
    center_lonlat: tuple[float, float]


def utm_epsg_for_lonlat(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def transformers_for_coordinates(
    coordinates: list[tuple[float, float]],
) -> tuple[Transformer, Transformer, int]:
    lon = sum(point[0] for point in coordinates) / len(coordinates)
    lat = sum(point[1] for point in coordinates) / len(coordinates)
    epsg = utm_epsg_for_lonlat(lon, lat)
    to_metric = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    to_lonlat = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    return to_metric, to_lonlat, epsg


def polygon_from_closed_line(coordinates: list[tuple[float, float]]) -> Polygon:
    if len(coordinates) < 4 or coordinates[0] != coordinates[-1]:
        raise ValueError("Selected KML geometry is not a closed LineString.")
    polygon = Polygon(coordinates)
    if not polygon.is_valid:
        polygon = make_valid(polygon)
    if polygon.geom_type == "MultiPolygon":
        polygon = max(polygon.geoms, key=lambda geom: geom.area)
    if polygon.geom_type != "Polygon":
        polygon = polygon.buffer(0)
    if polygon.is_empty or polygon.geom_type != "Polygon":
        raise ValueError("Could not convert closed LineString to a valid polygon.")
    return polygon


def line_to_geojson(line: LineString) -> dict[str, Any]:
    return mapping(line)


def polygon_to_geojson(polygon: Polygon) -> dict[str, Any]:
    return mapping(polygon)


def stable_geometry_id(coordinates: list[tuple[float, float]]) -> str:
    digest = hashlib.sha1(repr(coordinates).encode("utf-8")).hexdigest()
    return digest[:12]


def interpolate_line(line: LineString, distance: float) -> tuple[float, float]:
    point = line.interpolate(distance)
    return point.x, point.y


def resample_line(line: LineString, step_m: float) -> list[tuple[float, float]]:
    if line.length <= 0:
        return list(line.coords)
    point_count = max(2, int(math.ceil(line.length / step_m)) + 1)
    distances = np.linspace(0, line.length, point_count)
    return [interpolate_line(line, float(distance)) for distance in distances]


def moving_average_points(
    points: list[tuple[float, float]],
    radius: int,
) -> list[tuple[float, float]]:
    if len(points) <= 2 or radius <= 0:
        return points

    smoothed: list[tuple[float, float]] = []
    for index, point in enumerate(points):
        if index == 0 or index == len(points) - 1:
            smoothed.append(point)
            continue

        start = max(0, index - radius)
        end = min(len(points), index + radius + 1)
        window = points[start:end]
        smoothed.append(
            (
                sum(candidate[0] for candidate in window) / len(window),
                sum(candidate[1] for candidate in window) / len(window),
            )
        )
    return smoothed


def smooth_metric_centerline(
    line: LineString,
    polygon: Polygon,
    pixel_size_m: float,
) -> LineString:
    simplified = line.simplify(pixel_size_m * 3.0, preserve_topology=False)
    if simplified.geom_type != "LineString" or len(simplified.coords) < 2:
        simplified = line

    sample_step_m = max(pixel_size_m * 2.0, 50.0)
    points = resample_line(simplified, sample_step_m)
    radius = max(1, int(round(150.0 / sample_step_m)))
    smoothed_points = moving_average_points(points, radius)
    inside_area = polygon.buffer(pixel_size_m)
    safe_points = [
        candidate if inside_area.covers(Point(candidate)) else original
        for original, candidate in zip(points, smoothed_points)
    ]

    smoothed = LineString(safe_points).simplify(pixel_size_m * 1.5, preserve_topology=False)
    if smoothed.geom_type != "LineString" or len(smoothed.coords) < 2:
        return simplified
    return smoothed


def longest_path_in_skeleton(mask: np.ndarray) -> list[tuple[int, int]]:
    rows, cols = np.nonzero(mask)
    nodes = set(zip(rows.tolist(), cols.tolist()))
    if not nodes:
        return []

    neighbors_cache: dict[tuple[int, int], list[tuple[tuple[int, int], float]]] = {}
    directions = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]

    def neighbors(node: tuple[int, int]) -> list[tuple[tuple[int, int], float]]:
        if node in neighbors_cache:
            return neighbors_cache[node]
        row, col = node
        found: list[tuple[tuple[int, int], float]] = []
        for d_row, d_col in directions:
            other = (row + d_row, col + d_col)
            if other in nodes:
                found.append((other, math.hypot(d_row, d_col)))
        neighbors_cache[node] = found
        return found

    def component(start: tuple[int, int]) -> set[tuple[int, int]]:
        seen = {start}
        stack = [start]
        while stack:
            node = stack.pop()
            for other, _weight in neighbors(node):
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
        return seen

    unseen = set(nodes)
    largest: set[tuple[int, int]] = set()
    while unseen:
        start = min(unseen)
        comp = component(start)
        if len(comp) > len(largest):
            largest = comp
        unseen -= comp

    def dijkstra(
        start: tuple[int, int],
    ) -> tuple[tuple[int, int], dict[tuple[int, int], tuple[int, int] | None]]:
        import heapq

        distances = {start: 0.0}
        previous: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        heap: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
        farthest = start

        while heap:
            distance, node = heapq.heappop(heap)
            if distance != distances[node]:
                continue
            if distance > distances[farthest] or (
                distance == distances[farthest] and node < farthest
            ):
                farthest = node
            for other, weight in neighbors(node):
                if other not in largest:
                    continue
                candidate = distance + weight
                if candidate < distances.get(other, float("inf")):
                    distances[other] = candidate
                    previous[other] = node
                    heapq.heappush(heap, (candidate, other))
        return farthest, previous

    first = min(largest)
    endpoint_a, _ = dijkstra(first)
    endpoint_b, previous = dijkstra(endpoint_a)

    path = [endpoint_b]
    while path[-1] != endpoint_a:
        prior = previous.get(path[-1])
        if prior is None:
            break
        path.append(prior)
    path.reverse()
    return path


def medial_axis_centerline(
    polygon_lonlat: Polygon,
    pixel_size_m: float = 25,
) -> tuple[LineString, int]:
    coordinates = list(polygon_lonlat.exterior.coords)
    to_metric, to_lonlat, epsg = transformers_for_coordinates(coordinates)
    polygon_m = transform(to_metric.transform, polygon_lonlat)
    min_x, min_y, max_x, max_y = polygon_m.bounds

    width = max(3, int(math.ceil((max_x - min_x) / pixel_size_m)) + 6)
    height = max(3, int(math.ceil((max_y - min_y) / pixel_size_m)) + 6)

    exterior = np.asarray(polygon_m.exterior.coords)
    cols = ((exterior[:, 0] - min_x) / pixel_size_m + 3).astype(float)
    rows = ((max_y - exterior[:, 1]) / pixel_size_m + 3).astype(float)

    mask = np.zeros((height, width), dtype=bool)
    rr, cc = draw_polygon(rows, cols, shape=mask.shape)
    mask[rr, cc] = True

    skeleton = medial_axis(mask, rng=0)
    path = longest_path_in_skeleton(skeleton)
    if len(path) < 2:
        raise ValueError("Could not build a centerline from polygon skeleton.")

    metric_points = [
        (
            min_x + (col - 3 + 0.5) * pixel_size_m,
            max_y - (row - 3 + 0.5) * pixel_size_m,
        )
        for row, col in path
    ]
    line_m = smooth_metric_centerline(LineString(metric_points), polygon_m, pixel_size_m)
    line_lonlat = transform(to_lonlat.transform, line_m)
    return line_lonlat, epsg


def square_aoi_for_segment(
    segment_m: LineString,
    context_m: float,
    to_lonlat: Transformer,
) -> tuple[Polygon, tuple[float, float]]:
    centroid = segment_m.interpolate(segment_m.length / 2)
    side_m = segment_m.length + 2 * context_m
    half = side_m / 2
    square_m = Polygon(
        [
            (centroid.x - half, centroid.y - half),
            (centroid.x + half, centroid.y - half),
            (centroid.x + half, centroid.y + half),
            (centroid.x - half, centroid.y + half),
            (centroid.x - half, centroid.y - half),
        ]
    )
    square_lonlat = transform(to_lonlat.transform, square_m)
    center_lonlat = to_lonlat.transform(centroid.x, centroid.y)
    return square_lonlat, center_lonlat


def build_segments(
    centerline_lonlat: LineString,
    segment_length_m: float,
    context_m: float,
) -> tuple[list[Segment], int]:
    coordinates = list(centerline_lonlat.coords)
    to_metric, to_lonlat, epsg = transformers_for_coordinates(coordinates)
    centerline_m = transform(to_metric.transform, centerline_lonlat)
    total_length = centerline_m.length

    segments: list[Segment] = []
    count = max(1, math.ceil(total_length / segment_length_m))
    for index in range(count):
        start_m = index * segment_length_m
        end_m = min((index + 1) * segment_length_m, total_length)
        if end_m - start_m < segment_length_m * 0.25 and segments:
            break

        sample_count = max(2, int(math.ceil((end_m - start_m) / 100)))
        distances = np.linspace(start_m, end_m, sample_count)
        segment_points = [interpolate_line(centerline_m, float(distance)) for distance in distances]
        segment_m = LineString(segment_points)
        segment_lonlat = transform(to_lonlat.transform, segment_m)
        aoi_lonlat, center_lonlat = square_aoi_for_segment(segment_m, context_m, to_lonlat)

        segments.append(
            Segment(
                id=f"seg_{index + 1:03d}",
                index=index + 1,
                start_m=start_m,
                end_m=end_m,
                length_m=end_m - start_m,
                line_lonlat=segment_lonlat,
                aoi_lonlat=aoi_lonlat,
                center_lonlat=center_lonlat,
            )
        )

    return segments, epsg


def bbox_from_polygon(polygon: Polygon) -> tuple[float, float, float, float]:
    min_x, min_y, max_x, max_y = polygon.bounds
    return min_x, min_y, max_x, max_y


def geometry_payload(
    kml_id: str,
    boundary_coordinates: list[tuple[float, float]],
    context_m: float,
    segment_length_m: float,
    skeleton_pixel_size_m: float = 25,
) -> dict[str, Any]:
    polygon = polygon_from_closed_line(boundary_coordinates)
    centerline, skeleton_epsg = medial_axis_centerline(
        polygon,
        pixel_size_m=skeleton_pixel_size_m,
    )
    segments, segment_epsg = build_segments(centerline, segment_length_m, context_m)

    return {
        "kml_id": kml_id,
        "source_geometry": "closed_linestring_boundary",
        "interpretation": "boundary_as_polygon",
        "centerline_method": "raster_medial_axis_approximation",
        "centerline_warning": "Approximate centerline, not a real pipeline geometry.",
        "centerline_smoothing": "metric simplify plus 150 m moving average",
        "skeleton_epsg": skeleton_epsg,
        "segment_epsg": segment_epsg,
        "segment_length_m": segment_length_m,
        "context_m": context_m,
        "polygon": polygon_to_geojson(polygon),
        "centerline": line_to_geojson(centerline),
        "segments": [
            {
                "id": segment.id,
                "index": segment.index,
                "start_m": round(segment.start_m, 2),
                "end_m": round(segment.end_m, 2),
                "length_m": round(segment.length_m, 2),
                "center": segment.center_lonlat,
                "line": line_to_geojson(segment.line_lonlat),
                "aoi": polygon_to_geojson(segment.aoi_lonlat),
                "bbox": bbox_from_polygon(segment.aoi_lonlat),
            }
            for segment in segments
        ],
    }

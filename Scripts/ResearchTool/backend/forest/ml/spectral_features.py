from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import rasterio


BAND_ORDER = ["B02", "B03", "B04", "B08", "B11", "B12", "SCL", "dataMask"]
INDEX_DEFS = {"ndvi": ("B08", "B04"), "nbr": ("B08", "B12"), "ndmi": ("B08", "B11")}
SCL_BAD = {0, 1, 3, 8, 9, 10, 11}
SCL_REPORT = (2, 4, 5, 6, 7)
REQUIRED_INDEX_MEDIANS = ("ndvi_median", "nbr_median", "ndmi_median")


def missing_required_index_medians(stats: dict[str, float]) -> list[str]:
    return [
        name
        for name in REQUIRED_INDEX_MEDIANS
        if name not in stats or not np.isfinite(float(stats[name]))
    ]


def spectral_input_status(pre_count: int, post_count: int) -> str:
    if pre_count == 0 and post_count == 0:
        return "NO_SPECTRAL_BOTH"
    if pre_count == 0:
        return "NO_SPECTRAL_PRE"
    if post_count == 0:
        return "NO_SPECTRAL_POST"
    return "READY"


def stats_for_tif(tif_path: str | Path) -> dict[str, float]:
    path = Path(tif_path)
    if not path.exists():
        return {}
    with rasterio.open(path) as source:
        if source.count != len(BAND_ORDER):
            return {}
        data = {
            name: source.read(index + 1).astype("float32")
            for index, name in enumerate(BAND_ORDER)
        }

    scl = data["SCL"].astype("int32")
    valid = (data["dataMask"] > 0) & ~np.isin(scl, list(SCL_BAD))
    result: dict[str, float] = {}
    for name, (first_band, second_band) in INDEX_DEFS.items():
        first = data[first_band]
        second = data[second_band]
        denominator = first + second
        usable = valid & np.isfinite(first) & np.isfinite(second) & (denominator != 0)
        if not usable.any():
            continue
        values = (first[usable] - second[usable]) / denominator[usable]
        result.update(
            {
                f"{name}_mean": float(values.mean()),
                f"{name}_median": float(np.median(values)),
                f"{name}_std": float(values.std()),
                f"{name}_p10": float(np.percentile(values, 10)),
                f"{name}_p90": float(np.percentile(values, 90)),
                f"{name}_valid_frac": float(usable.mean()),
            }
        )
    total = scl.size
    for scl_class in SCL_REPORT:
        result[f"scl_{scl_class}_frac"] = float((scl == scl_class).sum() / total)
    return result


def median_stats(paths: Iterable[str | Path]) -> dict[str, float]:
    return median_feature_rows(row for path in paths if (row := stats_for_tif(path)))


def median_feature_rows(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    rows = list(rows)
    keys = sorted({key for row in rows for key in row})
    return {
        key: float(np.median([row[key] for row in rows if key in row]))
        for key in keys
    }


def pre_post_spectral_features(
    pre_paths: Iterable[str | Path],
    post_paths: Iterable[str | Path],
) -> dict[str, float]:
    return pre_post_features_from_stats(median_stats(pre_paths), median_stats(post_paths))


def pre_post_features_from_stats(
    pre: dict[str, float],
    post: dict[str, float],
) -> dict[str, float]:
    result = {f"{key}_pre": value for key, value in pre.items()}
    result.update({f"{key}_post": value for key, value in post.items()})
    for key in sorted(set(pre) & set(post)):
        delta = post[key] - pre[key]
        result[f"{key}_delta"] = float(delta)
        result[f"{key}_rel"] = float(delta / (abs(pre[key]) + 1e-6))
    return result

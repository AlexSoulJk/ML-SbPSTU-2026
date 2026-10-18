from __future__ import annotations

import sys
from pathlib import Path

_here = Path(__file__).resolve()
for _p in _here.parents:
    if (_p / "backend").exists():
        sys.path.insert(0, str(_p))
        break

import numpy as np
import rasterio
from scipy import ndimage
from skimage.measure import label, regionprops


def loss_shape_features(
    loss_mask_path: Path,
    event_year: int | None = None,
    pixel_size_m: float | None = None,
) -> dict[str, float]:
    """Геометрические признаки маски потери леса для одного sample.

    Если в растре годовые коды Hansen (2..25) — фильтруем по event_year.
    Если растр бинарный (0/1) — берём всё, что > 0.
    """
    if not loss_mask_path.exists():
        return {}

    with rasterio.open(loss_mask_path) as src:
        arr = src.read(1).astype("int32")
        if pixel_size_m is None:
            pixel_size_m = abs(src.transform.a) * 111_000 if src.transform.a else 30.0

    arr_max = int(arr.max()) if arr.size else 0

    # Автоопределение: годовые коды Hansen или бинарная маска
    if event_year is not None and 2 <= arr_max <= 30:
        # lossyear-подобный растр: пиксель = год - 2000
        loss_code = event_year - 2000
        mask = (arr == loss_code)
    else:
        # бинарная маска доминирующего года / all_loss
        mask = (arr > 0)

    total_loss_px = int(mask.sum())
    if total_loss_px == 0:
        return {}

    px_area_ha = (pixel_size_m ** 2) / 10_000.0

    labeled = label(mask, connectivity=2)
    props = regionprops(labeled)
    if not props:
        return {}

    areas_px = np.array([p.area for p in props], dtype=float)
    areas_ha = areas_px * px_area_ha

    largest_share = float(areas_px.max() / total_loss_px)

    compactness_values = []
    rectangularity_values = []
    perimeter_area_values = []
    for p in props:
        per = p.perimeter if p.perimeter > 0 else 1e-6
        compactness_values.append(4 * np.pi * p.area / (per ** 2))

        minr, minc, maxr, maxc = p.bbox
        bbox_area = max((maxr - minr) * (maxc - minc), 1)
        rectangularity_values.append(p.area / bbox_area)

        perimeter_area_values.append(per / max(p.area, 1))

    weights = areas_px / areas_px.sum()
    compactness = float(np.average(compactness_values, weights=weights))
    rectangularity = float(np.average(rectangularity_values, weights=weights))
    perimeter_area = float(np.average(perimeter_area_values, weights=weights))

    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3)))
    edge_px = int(mask.sum() - eroded.sum())
    edge_density = float(edge_px / total_loss_px)

    return {
        "loss_total_area_ha": float(total_loss_px * px_area_ha),
        "loss_n_components": int(len(props)),
        "loss_largest_share": largest_share,
        "loss_mean_area_ha": float(areas_ha.mean()),
        "loss_compactness": compactness,
        "loss_rectangularity": rectangularity,
        "loss_perimeter_area_ratio": perimeter_area,
        "loss_edge_density": edge_density,
    }
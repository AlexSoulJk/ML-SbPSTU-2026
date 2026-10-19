from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def _effective_n_splits(labels: list[str], groups: list[str] | None, requested: int) -> int:
    if groups is None:
        counts = Counter(labels)
    else:
        labels_by_group: dict[str, set[str]] = defaultdict(set)
        for label, group in zip(labels, groups, strict=True):
            labels_by_group[group].add(label)
        invalid = [group for group, values in labels_by_group.items() if len(values) != 1]
        if invalid:
            raise ValueError(f"Groups contain multiple labels: {invalid[:5]}")
        counts = Counter(next(iter(values)) for values in labels_by_group.values())
    if len(counts) < 2:
        raise ValueError("At least two classes are required for stratified splits.")
    n_splits = min(int(requested), min(counts.values()))
    if n_splits < 2:
        raise ValueError(f"Not enough examples/groups per class for CV: {dict(counts)}")
    return n_splits


def build_saved_splits(
    records: list[dict[str, Any]],
    *,
    requested_n_splits: int,
    seeds: list[int],
    include_ordinary: bool = True,
) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot build splits for an empty record set.")
    record_ids = np.asarray([str(row["record_id"]) for row in records], dtype=object)
    labels = np.asarray([str(row["label"]) for row in records], dtype=object)
    groups = np.asarray([str(row["sample_id"]) for row in records], dtype=object)
    indices = np.arange(len(records))
    rows: list[dict[str, Any]] = []

    group_n_splits = _effective_n_splits(labels.tolist(), groups.tolist(), requested_n_splits)
    ordinary_n_splits = _effective_n_splits(labels.tolist(), None, requested_n_splits)
    for seed in seeds:
        group_cv = StratifiedGroupKFold(
            n_splits=group_n_splits,
            shuffle=True,
            random_state=int(seed),
        )
        for fold, (train_index, test_index) in enumerate(
            group_cv.split(indices, labels, groups),
            start=1,
        ):
            rows.append(
                {
                    "strategy": "stratified_group",
                    "seed": int(seed),
                    "fold": fold,
                    "train_record_ids": record_ids[train_index].tolist(),
                    "test_record_ids": record_ids[test_index].tolist(),
                    "train_groups": sorted(set(groups[train_index].tolist())),
                    "test_groups": sorted(set(groups[test_index].tolist())),
                }
            )

        if include_ordinary:
            ordinary_cv = StratifiedKFold(
                n_splits=ordinary_n_splits,
                shuffle=True,
                random_state=int(seed),
            )
            for fold, (train_index, test_index) in enumerate(
                ordinary_cv.split(indices, labels),
                start=1,
            ):
                rows.append(
                    {
                        "strategy": "stratified",
                        "seed": int(seed),
                        "fold": fold,
                        "train_record_ids": record_ids[train_index].tolist(),
                        "test_record_ids": record_ids[test_index].tolist(),
                        "train_groups": sorted(set(groups[train_index].tolist())),
                        "test_groups": sorted(set(groups[test_index].tolist())),
                    }
                )

    return {
        "requested_n_splits": int(requested_n_splits),
        "group_n_splits": group_n_splits,
        "ordinary_n_splits": ordinary_n_splits,
        "seeds": [int(seed) for seed in seeds],
        "record_count": len(records),
        "group_count": len(set(groups.tolist())),
        "rows": rows,
    }


def assert_no_group_leakage(saved_splits: dict[str, Any]) -> None:
    for row in saved_splits.get("rows", []):
        if row.get("strategy") != "stratified_group":
            continue
        overlap = set(row["train_groups"]) & set(row["test_groups"])
        if overlap:
            raise AssertionError(
                f"Group leakage in seed={row['seed']} fold={row['fold']}: {sorted(overlap)[:5]}"
            )


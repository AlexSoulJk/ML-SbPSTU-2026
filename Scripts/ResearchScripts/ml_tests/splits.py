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
        # A multi-event point may legitimately contain events with different
        # human classes. StratifiedGroupKFold keeps the complete point in one
        # fold; the feasibility bound is therefore the number of groups in
        # which each class is represented, not a one-label-per-group rule.
        counts = Counter(
            label
            for values in labels_by_group.values()
            for label in values
        )
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


def build_constrained_grouped_splits(
    records: list[dict[str, Any]],
    *,
    old_record_ids: set[str],
    required_old_record_id_sets: dict[str, set[str]] | None = None,
    requested_n_splits: int,
    seeds: list[int],
    max_attempts_per_seed: int = 500,
) -> dict[str, Any]:
    """Build reproducible grouped folds valid for both FULL and OLD train cohorts.

    The full cohort can be well stratified while a small nested old cohort loses
    one class in a particular training fold. For each requested repeat seed we
    therefore search a deterministic sequence of effective seeds and persist
    both values in the split manifest.
    """

    by_id = {str(row["record_id"]): row for row in records}
    old_sets = required_old_record_id_sets or {"old": old_record_ids}
    accepted_rows: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    used_effective_seeds: set[int] = set()
    for repeat_index, requested_seed in enumerate(seeds):
        accepted = None
        for attempt in range(max_attempts_per_seed):
            effective_seed = (
                int(requested_seed)
                if attempt == 0
                else 100_000 + repeat_index * max_attempts_per_seed + attempt
            )
            if effective_seed in used_effective_seeds:
                continue
            candidate = build_saved_splits(
                records,
                requested_n_splits=requested_n_splits,
                seeds=[effective_seed],
                include_ordinary=False,
            )
            assert_no_group_leakage(candidate)
            reasons: list[str] = []
            for row in candidate["rows"]:
                test = [by_id[str(item)] for item in row["test_record_ids"]]
                full_train = [by_id[str(item)] for item in row["train_record_ids"]]
                if len({item["label"] for item in test}) < 2:
                    reasons.append(f"fold_{row['fold']}:test_one_class")
                if len({item["label"] for item in full_train}) < 2:
                    reasons.append(f"fold_{row['fold']}:full_train_one_class")
                for cohort_name, cohort_ids in sorted(old_sets.items()):
                    old_train = [
                        by_id[str(item)]
                        for item in row["train_record_ids"]
                        if str(item) in cohort_ids
                    ]
                    if len({item["label"] for item in old_train}) < 2:
                        reasons.append(
                            f"fold_{row['fold']}:{cohort_name}_old_train_one_class"
                        )
            attempts.append(
                {
                    "requested_seed": int(requested_seed),
                    "effective_seed": effective_seed,
                    "attempt": attempt,
                    "accepted": not reasons,
                    "reasons": reasons,
                }
            )
            if reasons:
                continue
            for row in candidate["rows"]:
                row["requested_seed"] = int(requested_seed)
                row["effective_seed"] = effective_seed
                row["repeat_index"] = repeat_index
            accepted = candidate
            used_effective_seeds.add(effective_seed)
            break
        if accepted is None:
            raise ValueError(
                f"Could not build class-complete grouped folds for requested seed "
                f"{requested_seed} after {max_attempts_per_seed} attempts."
            )
        accepted_rows.extend(accepted["rows"])

    return {
        "requested_n_splits": int(requested_n_splits),
        "group_n_splits": len(accepted_rows) // len(seeds),
        "ordinary_n_splits": None,
        "requested_seeds": [int(seed) for seed in seeds],
        "effective_seeds": sorted(used_effective_seeds),
        "record_count": len(records),
        "group_count": len({str(row["sample_id"]) for row in records}),
        "old_record_count": len(old_record_ids & set(by_id)),
        "old_record_counts": {
            name: len(record_ids & set(by_id))
            for name, record_ids in sorted(old_sets.items())
        },
        "rows": accepted_rows,
        "attempts": attempts,
    }


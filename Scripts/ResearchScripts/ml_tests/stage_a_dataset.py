from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ResearchTool.backend.forest.models import (
    DerivedPreview,
    EventLabel,
    EventReview,
    HansenAnalysis,
    ManualValidation,
    Sample,
    SampleEvent,
    SampleEventDownload,
    SentinelDownload,
    SentinelSceneReview,
    SentinelSceneReviewReason,
)
from ResearchTool.backend.forest.storage_paths import resolve_storage_path, to_storage_path

from .config import PROJECT_ROOT, stable_json_hash
from .dataset import file_sha256


TARGET_CLASSES = ("Logging", "Wildfire")
PERIODS = ("PRE", "POST")


def _iso(value: Any) -> str:
    return value.isoformat() if value is not None else ""


def _json_object(value: str | None) -> dict[str, Any]:
    try:
        payload = json.loads(value or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _json_list(value: str | None) -> list[dict[str, Any]]:
    try:
        payload = json.loads(value or "[]")
    except (TypeError, json.JSONDecodeError):
        return []
    return [row for row in payload if isinstance(row, dict)] if isinstance(payload, list) else []


def _portable_path(path: str | Path | None) -> str:
    if not path:
        return ""
    resolved = resolve_storage_path(path).resolve(strict=False)
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _path_exists(path: str | Path | None) -> bool:
    return bool(path) and resolve_storage_path(path).exists()


def _latest_by_download(rows: Iterable[Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for row in rows:
        result.setdefault(str(row.download_id), row)
    return result


def _hansen_context(
    sample: Sample,
    events: list[SampleEvent],
    analysis: HansenAnalysis | None,
) -> dict[str, float]:
    raw = _json_object(sample.raw_properties_json)
    metrics = raw.get("candidate_metrics") or {}
    total_area = float(metrics.get("total_loss_area_ha") or 0.0)
    loss_fraction = float(metrics.get("post_2016_loss_aoi_percent") or 0.0) / 100.0

    if analysis is not None:
        if total_area <= 0:
            total_area = float(analysis.total_loss_area_ha or 0.0)
        if loss_fraction <= 0 and analysis.plot_size_m:
            plot_area_ha = float(analysis.plot_size_m) ** 2 / 10_000.0
            loss_fraction = total_area / plot_area_ha if plot_area_ha else 0.0

    event_areas = [float(event.loss_area_ha or 0.0) for event in events]
    if total_area <= 0:
        total_area = sum(event_areas)
    dominant_share = max(event_areas, default=0.0) / total_area if total_area else 0.0
    max_year = max((event.event_year for event in events), default=0)
    recent_area = sum(
        float(event.loss_area_ha or 0.0)
        for event in events
        if event.event_year > max_year - 5
    )
    recent_fraction = loss_fraction * recent_area / total_area if total_area else 0.0
    return {
        "hansen_loss_fraction": float(loss_fraction),
        "hansen_recent_loss_5y": float(recent_fraction),
        "hansen_dominant_share": float(dominant_share),
        "hansen_total_loss_area_ha": float(total_area),
    }


def build_current_event_manifest(
    db: Session,
    *,
    database_fingerprint: dict[str, Any],
    target_classes: Iterable[str] = TARGET_CLASSES,
    virtual_links: Iterable[dict[str, Any]] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build the complete immutable human-VALID event manifest.

    Event-level reviews are authoritative. Older sample-level manual VALID
    labels are inherited by their events only when the event has no explicit
    review of its own.
    """

    classes = tuple(sorted(set(str(value) for value in target_classes)))
    reviewed_rows = db.execute(
        select(EventReview, EventLabel, SampleEvent, Sample)
        .join(EventLabel, EventLabel.label_id == EventReview.human_label_id)
        .join(SampleEvent, SampleEvent.event_id == EventReview.event_id)
        .join(Sample, Sample.sample_id == SampleEvent.sample_id)
        .where(
            func.upper(func.trim(EventReview.review_status)) == "VALID",
            EventLabel.label.in_(classes),
        )
        .order_by(SampleEvent.sample_id, SampleEvent.event_year, SampleEvent.event_id)
    ).all()
    explicitly_reviewed_event_ids = set(db.scalars(select(EventReview.event_id)))
    legacy_rows = db.execute(
        select(ManualValidation, SampleEvent, Sample)
        .join(Sample, Sample.sample_id == ManualValidation.sample_id)
        .join(SampleEvent, SampleEvent.sample_id == Sample.sample_id)
        .where(
            func.upper(func.trim(ManualValidation.validation)) == "VALID",
            Sample.driver_primary.in_(classes),
        )
        .order_by(SampleEvent.sample_id, SampleEvent.event_year, SampleEvent.event_id)
    ).all()

    ground_truth: list[dict[str, Any]] = []
    for review, human_label, event, sample in reviewed_rows:
        ground_truth.append(
            {
                "event": event,
                "sample": sample,
                "human_label": str(human_label.label),
                "ground_truth_source": "EVENT_REVIEW",
                "ground_truth_id": str(event.event_id),
                "review_id": str(event.event_id),
                "reviewer_name": str(review.reviewer_name or ""),
                "review_updated_at": review.updated_at,
                "snapshot": {
                    "event_id": str(event.event_id),
                    "review_status": str(review.review_status),
                    "human_label_id": str(review.human_label_id or ""),
                    "human_label": str(human_label.label),
                    "reviewer_name": str(review.reviewer_name or ""),
                    "updated_at": _iso(review.updated_at),
                },
            }
        )
    for validation, event, sample in legacy_rows:
        if str(event.event_id) in explicitly_reviewed_event_ids:
            continue
        ground_truth.append(
            {
                "event": event,
                "sample": sample,
                "human_label": str(sample.driver_primary),
                "ground_truth_source": "SAMPLE_MANUAL_VALIDATION",
                "ground_truth_id": str(sample.sample_id),
                "review_id": f"manual:{sample.sample_id}",
                "reviewer_name": "",
                "review_updated_at": validation.updated_at,
                "snapshot": {
                    "sample_id": str(sample.sample_id),
                    "validation": str(validation.validation),
                    "human_label": str(sample.driver_primary),
                    "updated_at": _iso(validation.updated_at),
                },
            }
        )
    ground_truth.sort(
        key=lambda row: (
            str(row["sample"].sample_id),
            int(row["event"].event_year),
            str(row["event"].event_id),
        )
    )
    if not ground_truth:
        return [], [], {"events": 0, "reason": "no_valid_human_events"}

    event_ids = [str(row["event"].event_id) for row in ground_truth]
    sample_ids = sorted({str(row["event"].sample_id) for row in ground_truth})
    explicit_links = list(
        db.scalars(
            select(SampleEventDownload)
            .where(SampleEventDownload.event_id.in_(event_ids))
            .order_by(
                SampleEventDownload.event_id,
                SampleEventDownload.period,
                SampleEventDownload.download_id,
            )
        )
    )
    links_by_event: dict[str, list[dict[str, str]]] = defaultdict(list)
    explicitly_linked_download_ids: set[str] = set()
    for link in explicit_links:
        download_id = str(link.download_id)
        explicitly_linked_download_ids.add(download_id)
        links_by_event[str(link.event_id)].append(
            {
                "download_id": download_id,
                "period": str(link.period or "").upper(),
                "event_assignment": "EXPLICIT",
            }
        )

    event_id_set = set(event_ids)
    virtual_link_count = 0
    for raw_link in virtual_links:
        event_id = str(raw_link.get("event_id") or "")
        download_id = str(raw_link.get("download_id") or "")
        if (
            not event_id
            or event_id not in event_id_set
            or not download_id
            or download_id in explicitly_linked_download_ids
        ):
            continue
        links_by_event[event_id].append(
            {
                "download_id": download_id,
                "period": str(
                    raw_link.get("legacy_period") or raw_link.get("period") or ""
                ).upper(),
                "event_assignment": "LEGACY_DOMINANT_BACKFILL",
            }
        )
        virtual_link_count += 1

    download_ids = sorted(
        {
            link["download_id"]
            for event_links in links_by_event.values()
            for link in event_links
        }
    )
    downloads = {
        str(row.download_id): row
        for row in db.scalars(
            select(SentinelDownload).where(SentinelDownload.download_id.in_(download_ids))
        )
    }
    previews = _latest_by_download(
        db.scalars(
            select(DerivedPreview)
            .where(
                DerivedPreview.download_id.in_(download_ids),
                DerivedPreview.view == "rgb",
            )
            .order_by(DerivedPreview.download_id, DerivedPreview.created_at.desc())
        )
    )
    scene_reviews = {
        str(row.download_id): row
        for row in db.scalars(
            select(SentinelSceneReview).where(SentinelSceneReview.download_id.in_(download_ids))
        )
    }
    reason_codes: dict[str, list[str]] = defaultdict(list)
    for reason in db.scalars(
        select(SentinelSceneReviewReason)
        .where(SentinelSceneReviewReason.download_id.in_(download_ids))
        .order_by(SentinelSceneReviewReason.download_id, SentinelSceneReviewReason.id)
    ):
        code = str(reason.reason_code)
        if code not in reason_codes[str(reason.download_id)]:
            reason_codes[str(reason.download_id)].append(code)

    all_events_by_sample: dict[str, list[SampleEvent]] = defaultdict(list)
    for event in db.scalars(
        select(SampleEvent)
        .where(SampleEvent.sample_id.in_(sample_ids))
        .order_by(SampleEvent.sample_id, SampleEvent.event_year, SampleEvent.event_id)
    ):
        all_events_by_sample[str(event.sample_id)].append(event)

    latest_analysis: dict[str, HansenAnalysis] = {}
    for analysis in db.scalars(
        select(HansenAnalysis)
        .where(HansenAnalysis.sample_id.in_(sample_ids))
        .order_by(HansenAnalysis.sample_id, HansenAnalysis.created_at.desc())
    ):
        latest_analysis.setdefault(str(analysis.sample_id), analysis)

    event_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    exclusion_counts: Counter[str] = Counter()
    for truth in ground_truth:
        event = truth["event"]
        sample = truth["sample"]
        human_label = str(truth["human_label"])
        sample_events = all_events_by_sample[str(sample.sample_id)]
        context = _hansen_context(sample, sample_events, latest_analysis.get(str(sample.sample_id)))
        inputs: list[dict[str, Any]] = []
        for link in links_by_event.get(str(event.event_id), []):
            period = str(link["period"] or "").upper()
            if period not in PERIODS:
                continue
            download = downloads.get(str(link["download_id"]))
            if download is None:
                exclusion_counts["MISSING_DOWNLOAD_ROW"] += 1
                continue
            preview = previews.get(str(download.download_id))
            scene_review = scene_reviews.get(str(download.download_id))
            exclusions: list[str] = []
            if bool(download.is_bad_cloud):
                exclusions.append("BAD_CLOUD")
            if bool(download.is_bad_quality):
                exclusions.append("BAD_QUALITY")
            if scene_review is not None and bool(scene_review.is_excluded):
                exclusions.append("SCENE_REVIEW_EXCLUDED")
            if not _path_exists(download.local_path):
                exclusions.append("MISSING_GEOTIFF")
            rgb_exists = preview is not None and _path_exists(preview.png_path)
            for code in exclusions:
                exclusion_counts[code] += 1
            base_usable = not any(
                code in {"BAD_CLOUD", "BAD_QUALITY", "SCENE_REVIEW_EXCLUDED"}
                for code in exclusions
            )
            input_row = {
                "download_id": str(download.download_id),
                "scene_id": str(download.scene_id or ""),
                "period": period,
                "event_assignment": link["event_assignment"],
                "geotiff_path": _portable_path(download.local_path),
                "rgb_path": _portable_path(preview.png_path if preview else None),
                "rgb_metadata_path": _portable_path(preview.metadata_path if preview else None),
                "cloud_fraction": download.cloud_fraction,
                "shadow_fraction": download.shadow_fraction,
                "nodata_fraction": download.nodata_fraction,
                "dark_fraction": download.dark_fraction,
                "is_bad_cloud": bool(download.is_bad_cloud),
                "is_bad_quality": bool(download.is_bad_quality),
                "scene_review_excluded": bool(scene_review.is_excluded) if scene_review else False,
                "scene_review_reasons": reason_codes.get(str(download.download_id), []),
                "exclusion_reasons": exclusions,
                "xgboost_usable": base_usable and "MISSING_GEOTIFF" not in exclusions,
                "remoteclip_usable": base_usable and rgb_exists,
            }
            inputs.append(input_row)
            if input_row["remoteclip_usable"]:
                image_rows.append(
                    {
                        "record_id": str(event.event_id),
                        "sample_id": str(sample.sample_id),
                        "sample_source_file": str(sample.source_file or ""),
                        "sample_source_format": str(sample.source_format or ""),
                        "event_id": str(event.event_id),
                        "event_year": int(event.event_year),
                        "event_loss_area_ha": event.loss_area_ha,
                        "event_loss_share": event.loss_share,
                        "event_is_dominant": bool(event.is_dominant),
                        "is_multi_event_area": bool(sample.has_multiple_events),
                        "label": human_label,
                        "validation_cohort": "MANUAL_VALID",
                        "ground_truth_source": truth["ground_truth_source"],
                        "download_id": str(download.download_id),
                        "period": period,
                        "view": "rgb",
                        "png_path": input_row["rgb_path"],
                        "eligible": True,
                    }
                )

        period_counts = {
            model: {
                period: sum(bool(row[f"{model}_usable"]) and row["period"] == period for row in inputs)
                for period in PERIODS
            }
            for model in ("xgboost", "remoteclip")
        }
        event_rows.append(
            {
                "record_id": str(event.event_id),
                "sample_id": str(sample.sample_id),
                "sample_source_file": str(sample.source_file or ""),
                "sample_source_format": str(sample.source_format or ""),
                "event_id": str(event.event_id),
                "event_year": int(event.event_year),
                "event_source": str(event.source or ""),
                "event_loss_area_ha": event.loss_area_ha,
                "event_loss_share": event.loss_share,
                "event_is_dominant": bool(event.is_dominant),
                "is_multi_event_area": bool(sample.has_multiple_events),
                "human_label": human_label,
                "ground_truth_source": truth["ground_truth_source"],
                "ground_truth_id": truth["ground_truth_id"],
                "review_id": truth["review_id"],
                "review_snapshot_id": stable_json_hash(truth["snapshot"]),
                "review_updated_at": _iso(truth["review_updated_at"]),
                "reviewer_name": truth["reviewer_name"],
                "xgboost_input_status": (
                    "READY"
                    if all(period_counts["xgboost"][period] > 0 for period in PERIODS)
                    else "MISSING_PRE_OR_POST"
                ),
                "remoteclip_input_status": (
                    "READY"
                    if all(period_counts["remoteclip"][period] > 0 for period in PERIODS)
                    else "MISSING_PRE_OR_POST"
                ),
                "period_counts": period_counts,
                "inputs": sorted(inputs, key=lambda row: (row["period"], row["download_id"])),
                **context,
            }
        )

    event_rows.sort(key=lambda row: (row["sample_id"], row["event_year"], row["event_id"]))
    image_rows.sort(key=lambda row: (row["sample_id"], row["event_year"], row["period"], row["download_id"]))
    hash_rows = [
        {
            "event_id": row["event_id"],
            "sample_id": row["sample_id"],
            "event_year": row["event_year"],
            "human_label": row["human_label"],
            "ground_truth_source": row["ground_truth_source"],
            "review_snapshot_id": row["review_snapshot_id"],
            "xgboost_input_status": row["xgboost_input_status"],
            "remoteclip_input_status": row["remoteclip_input_status"],
            "inputs": [
                {
                    "download_id": item["download_id"],
                    "period": item["period"],
                    "event_assignment": item["event_assignment"],
                    "xgboost_usable": item["xgboost_usable"],
                    "remoteclip_usable": item["remoteclip_usable"],
                    "exclusion_reasons": item["exclusion_reasons"],
                }
                for item in row["inputs"]
            ],
        }
        for row in event_rows
    ]
    audit = {
        "events": len(event_rows),
        "samples": len({row["sample_id"] for row in event_rows}),
        "class_counts": dict(sorted(Counter(row["human_label"] for row in event_rows).items())),
        "ground_truth_source_counts": dict(
            sorted(Counter(row["ground_truth_source"] for row in event_rows).items())
        ),
        "ground_truth_source_sample_counts": {
            source: len(
                {
                    row["sample_id"]
                    for row in event_rows
                    if row["ground_truth_source"] == source
                }
            )
            for source in sorted({row["ground_truth_source"] for row in event_rows})
        },
        "xgboost_ready_events": sum(row["xgboost_input_status"] == "READY" for row in event_rows),
        "remoteclip_ready_events": sum(
            row["remoteclip_input_status"] == "READY" for row in event_rows
        ),
        "raw_image_rows": len(image_rows),
        "explicit_event_download_links": len(explicit_links),
        "virtual_dominant_backfill_links": virtual_link_count,
        "input_exclusion_counts": dict(sorted(exclusion_counts.items())),
        "database_fingerprint": database_fingerprint,
        "qc_snapshot_hash": stable_json_hash(hash_rows),
        "manifest_hash": stable_json_hash(hash_rows),
    }
    return event_rows, image_rows, audit


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _current_event_lookup(db: Session) -> dict[tuple[str, int], list[str]]:
    result: dict[tuple[str, int], list[str]] = defaultdict(list)
    for event in db.scalars(select(SampleEvent).order_by(SampleEvent.event_id)):
        result[(str(event.sample_id), int(event.event_year))].append(str(event.event_id))
    return result


def reconstruct_remoteclip_legacy_cohort(
    db: Session,
    *,
    training_output_path: Path,
    legacy_manifest: dict[str, Any],
) -> dict[str, Any]:
    rows = _read_csv(training_output_path)
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        if str(row.get("eligible") or "").lower() not in {"true", "1"}:
            continue
        if row.get("validation_cohort") != "MANUAL_VALID":
            continue
        item = grouped.setdefault(
            str(row["event_id"]),
            {
                "legacy_event_id": str(row["event_id"]),
                "sample_id": str(row["sample_id"]),
                "event_year": int(row["event_year"]),
                "periods": set(),
            },
        )
        item["periods"].add(str(row["period"]).upper())
    reconstructed = [row for row in grouped.values() if set(PERIODS) <= row["periods"]]
    legacy_ids = sorted(row["legacy_event_id"] for row in reconstructed)
    expected_hash = str(
        (legacy_manifest.get("training_cohort") or {}).get("record_ids_sha256") or ""
    )
    lookup = _current_event_lookup(db)
    mapped: list[dict[str, Any]] = []
    for row in reconstructed:
        current = lookup.get((row["sample_id"], row["event_year"]), [])
        mapped.append(
            {
                "legacy_event_id": row["legacy_event_id"],
                "sample_id": row["sample_id"],
                "event_year": row["event_year"],
                "current_event_id": current[0] if len(current) == 1 else "",
                "mapping_status": "EXACT" if len(current) == 1 else "MISSING" if not current else "AMBIGUOUS",
            }
        )
    actual_hash = stable_json_hash(legacy_ids)
    return {
        "source": _portable_path(training_output_path),
        "legacy_artifact_version": legacy_manifest.get("producer_version"),
        "expected_record_ids_sha256": expected_hash,
        "reconstructed_record_ids_sha256": actual_hash,
        "hash_matches_artifact": bool(expected_hash) and actual_hash == expected_hash,
        "legacy_records": len(legacy_ids),
        "mapped_current_events": sum(row["mapping_status"] == "EXACT" for row in mapped),
        "rows": mapped,
    }


def reconstruct_xgboost_legacy_cohort(
    db: Session,
    *,
    dataset_path: Path,
    legacy_manifest: dict[str, Any],
    target_classes: Iterable[str] = TARGET_CLASSES,
) -> dict[str, Any]:
    expected_hash = str((legacy_manifest.get("training") or {}).get("dataset_sha256") or "")
    actual_hash = file_sha256(dataset_path)
    frame = pd.read_parquet(dataset_path)
    frame = frame[frame["driver_primary"].isin(tuple(target_classes))].copy()
    lookup = _current_event_lookup(db)
    rows: list[dict[str, Any]] = []
    for row in frame.sort_values(["sample_id", "hansen_event_year"]).itertuples():
        key = (str(row.sample_id), int(row.hansen_event_year))
        current = lookup.get(key, [])
        rows.append(
            {
                "sample_id": key[0],
                "event_year": key[1],
                "legacy_label": str(row.driver_primary),
                "current_event_id": current[0] if len(current) == 1 else "",
                "mapping_status": "EXACT" if len(current) == 1 else "MISSING" if not current else "AMBIGUOUS",
            }
        )
    return {
        "source": _portable_path(dataset_path),
        "legacy_artifact_version": legacy_manifest.get("producer_version"),
        "expected_dataset_sha256": expected_hash,
        "reconstructed_dataset_sha256": actual_hash,
        "hash_matches_artifact": bool(expected_hash) and actual_hash == expected_hash,
        "legacy_records": len(rows),
        "mapped_current_events": sum(row["mapping_status"] == "EXACT" for row in rows),
        "rows": rows,
    }


def load_frozen_remoteclip_legacy_cohort(
    db: Session,
    *,
    cohort_path: Path,
    legacy_manifest: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Load the once-reconstructed, Git-tracked RemoteCLIP v1 cohort."""

    source_rows = _read_csv(cohort_path)
    frozen_keys = [
        {
            "legacy_event_id": str(row["legacy_event_id"]),
            "sample_id": str(row["sample_id"]),
            "event_year": int(row["event_year"]),
        }
        for row in source_rows
    ]
    metadata = dict(provenance.get("remoteclip") or {})
    if stable_json_hash(frozen_keys) != metadata.get("frozen_rows_sha256"):
        raise RuntimeError("Frozen RemoteCLIP legacy cohort hash mismatch")
    legacy_ids = sorted(row["legacy_event_id"] for row in frozen_keys)
    actual_record_hash = stable_json_hash(legacy_ids)
    expected_record_hash = str(
        (legacy_manifest.get("training_cohort") or {}).get("record_ids_sha256") or ""
    )
    if actual_record_hash != expected_record_hash:
        raise RuntimeError("Frozen RemoteCLIP cohort does not match the v1 artifact")

    lookup = _current_event_lookup(db)
    mapped: list[dict[str, Any]] = []
    for row in frozen_keys:
        current = lookup.get((row["sample_id"], row["event_year"]), [])
        mapped.append(
            {
                **row,
                "current_event_id": current[0] if len(current) == 1 else "",
                "mapping_status": (
                    "EXACT" if len(current) == 1 else "MISSING" if not current else "AMBIGUOUS"
                ),
            }
        )
    return {
        "source": _portable_path(cohort_path),
        "legacy_artifact_version": legacy_manifest.get("producer_version"),
        "expected_record_ids_sha256": expected_record_hash,
        "reconstructed_record_ids_sha256": actual_record_hash,
        "frozen_rows_sha256": metadata.get("frozen_rows_sha256"),
        "hash_matches_artifact": True,
        "legacy_records": len(mapped),
        "mapped_current_events": sum(row["mapping_status"] == "EXACT" for row in mapped),
        "rows": mapped,
    }


def load_frozen_xgboost_legacy_cohort(
    db: Session,
    *,
    cohort_path: Path,
    legacy_manifest: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Load the once-reconstructed, Git-tracked XGBoost v1 cohort."""

    source_rows = _read_csv(cohort_path)
    frozen_keys = [
        {
            "sample_id": str(row["sample_id"]),
            "event_year": int(row["event_year"]),
            "legacy_label": str(row["legacy_label"]),
        }
        for row in source_rows
    ]
    metadata = dict(provenance.get("xgboost") or {})
    if stable_json_hash(frozen_keys) != metadata.get("frozen_rows_sha256"):
        raise RuntimeError("Frozen XGBoost legacy cohort hash mismatch")
    expected_dataset_hash = str(
        (legacy_manifest.get("training") or {}).get("dataset_sha256") or ""
    )
    if metadata.get("artifact_source_dataset_sha256") != expected_dataset_hash:
        raise RuntimeError("Frozen XGBoost cohort provenance does not match the v1 artifact")

    lookup = _current_event_lookup(db)
    mapped: list[dict[str, Any]] = []
    for row in frozen_keys:
        current = lookup.get((row["sample_id"], row["event_year"]), [])
        mapped.append(
            {
                **row,
                "current_event_id": current[0] if len(current) == 1 else "",
                "mapping_status": (
                    "EXACT" if len(current) == 1 else "MISSING" if not current else "AMBIGUOUS"
                ),
            }
        )
    return {
        "source": _portable_path(cohort_path),
        "legacy_artifact_version": legacy_manifest.get("producer_version"),
        "expected_dataset_sha256": expected_dataset_hash,
        "reconstructed_dataset_sha256": metadata.get("artifact_source_dataset_sha256"),
        "frozen_rows_sha256": metadata.get("frozen_rows_sha256"),
        "hash_matches_artifact": True,
        "legacy_records": len(mapped),
        "mapped_current_events": sum(row["mapping_status"] == "EXACT" for row in mapped),
        "rows": mapped,
    }


def load_reviewed_baseline_cohort(
    baseline_path: Path,
    current_event_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    current = {str(row["event_id"]): row for row in current_event_rows}
    rows: list[dict[str, Any]] = []
    for row in _read_csv(baseline_path):
        if str(row.get("review_status") or "").upper() != "VALID":
            continue
        label = str(row.get("human_label") or "")
        if label not in TARGET_CLASSES:
            continue
        event_id = str(row.get("event_id") or "")
        current_row = current.get(event_id)
        status = "MATCHED"
        if current_row is None:
            status = "NOT_CURRENT_VALID"
        elif str(current_row["human_label"]) != label:
            status = "LABEL_CHANGED"
        rows.append(
            {
                "event_id": event_id,
                "sample_id": str(row.get("sample_id") or ""),
                "event_year": int(row.get("event_year") or 0),
                "baseline_label": label,
                "current_label": str(current_row["human_label"]) if current_row else "",
                "status": status,
            }
        )
    matched_ids = sorted(row["event_id"] for row in rows if row["status"] == "MATCHED")
    return {
        "source": _portable_path(baseline_path),
        "rows": rows,
        "matched_event_ids": matched_ids,
        "matched_event_ids_sha256": stable_json_hash(matched_ids),
        "status_counts": dict(sorted(Counter(row["status"] for row in rows).items())),
    }


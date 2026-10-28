from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import FOREST_MODELS_DIR
from .models import (
    DerivedPreview,
    EventLabel,
    SampleAssignment,
    SampleEvent,
    SampleEventDownload,
    SentinelDownload,
    SentinelSceneReview,
)
from .storage_paths import resolve_storage_path, to_storage_path


MODEL_SPECS = {
    "REMOTECLIP_LOGREG": {
        "directory": "remoteclip_logreg",
        "artifact": "classifier.joblib",
    },
    "XGBOOST": {
        "directory": "xgboost",
        "artifact": "model.json",
    },
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_version(version: str) -> str:
    value = version.strip()
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"Invalid model version: {version}")
    return value


def model_artifact_status(producer_name: str, version: str) -> dict[str, Any]:
    spec = MODEL_SPECS[producer_name]
    safe_version = _safe_version(version)
    model_dir = FOREST_MODELS_DIR / spec["directory"] / safe_version
    manifest_path = model_dir / "manifest.json"
    artifact_path = model_dir / spec["artifact"]
    errors: list[str] = []
    manifest = None
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"Invalid manifest.json: {exc}")
    else:
        errors.append("manifest.json is missing")
    artifact_sha256 = None
    if artifact_path.exists():
        artifact_sha256 = file_sha256(artifact_path)
        expected_hash = (manifest or {}).get("artifact_sha256")
        if expected_hash and expected_hash != artifact_sha256:
            errors.append("Artifact SHA-256 does not match manifest")
    else:
        errors.append(f"{spec['artifact']} is missing")
    return {
        "producer_name": producer_name,
        "producer_version": safe_version,
        "ready": not errors,
        "errors": errors,
        "model_dir": to_storage_path(model_dir) or str(model_dir),
        "manifest_path": to_storage_path(manifest_path) or str(manifest_path),
        "artifact_path": to_storage_path(artifact_path) or str(artifact_path),
        "artifact_sha256": artifact_sha256,
        "manifest": manifest,
    }


def _download_usable(
    download: SentinelDownload,
    review: SentinelSceneReview | None,
    rgb_preview: DerivedPreview | None,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not resolve_storage_path(download.local_path).exists():
        reasons.append("MISSING_GEOTIFF")
    if rgb_preview is None or not resolve_storage_path(rgb_preview.png_path).exists():
        reasons.append("MISSING_RGB_PREVIEW")
    if download.is_bad_cloud:
        reasons.append("AUTO_CLOUD_EXCLUDED")
    if download.is_bad_quality:
        reasons.append("AUTO_QUALITY_EXCLUDED")
    if review is not None and review.is_excluded:
        reasons.append("MANUAL_EXCLUDED")
    return not reasons, reasons


def build_model_labeling_plan(
    db: Session,
    *,
    mining_run_id: str,
    remoteclip_version: str = "v1",
    xgboost_version: str = "v1",
) -> dict[str, Any]:
    sample_ids = list(
        db.scalars(
            select(SampleAssignment.sample_id)
            .where(
                SampleAssignment.assignment_type == "hansen_mining_run",
                SampleAssignment.assignment_name == mining_run_id,
            )
            .order_by(SampleAssignment.sample_id)
        )
    )
    events = list(
        db.scalars(
            select(SampleEvent)
            .where(SampleEvent.sample_id.in_(sample_ids))
            .order_by(SampleEvent.sample_id, SampleEvent.event_year, SampleEvent.event_id)
        )
    ) if sample_ids else []
    event_ids = [event.event_id for event in events]
    links = list(
        db.scalars(
            select(SampleEventDownload).where(SampleEventDownload.event_id.in_(event_ids))
        )
    ) if event_ids else []
    download_ids = sorted({link.download_id for link in links})
    downloads = {
        row.download_id: row
        for row in db.scalars(
            select(SentinelDownload).where(SentinelDownload.download_id.in_(download_ids))
        )
    } if download_ids else {}
    reviews = {
        row.download_id: row
        for row in db.scalars(
            select(SentinelSceneReview).where(SentinelSceneReview.download_id.in_(download_ids))
        )
    } if download_ids else {}
    rgb_previews = {
        row.download_id: row
        for row in db.scalars(
            select(DerivedPreview).where(
                DerivedPreview.download_id.in_(download_ids),
                DerivedPreview.view == "rgb",
            )
        )
    } if download_ids else {}
    labels = list(
        db.scalars(
            select(EventLabel).where(
                EventLabel.event_id.in_(event_ids),
                EventLabel.source_type == "MODEL",
            )
        )
    ) if event_ids else []
    label_sources: dict[str, set[str]] = {}
    for label in labels:
        label_sources.setdefault(label.event_id, set()).add(label.source_name)

    links_by_event: dict[str, list[SampleEventDownload]] = {}
    for link in links:
        links_by_event.setdefault(link.event_id, []).append(link)
    rows: list[dict[str, Any]] = []
    for event in events:
        usable_by_period: dict[str, list[str]] = {"PRE": [], "EVENT": [], "POST": []}
        rejected: list[dict[str, Any]] = []
        for link in links_by_event.get(event.event_id, []):
            download = downloads.get(link.download_id)
            if download is None:
                rejected.append({"download_id": link.download_id, "reasons": ["MISSING_DB_DOWNLOAD"]})
                continue
            usable, reasons = _download_usable(
                download,
                reviews.get(download.download_id),
                rgb_previews.get(download.download_id),
            )
            if usable:
                usable_by_period.setdefault(link.period, []).append(download.download_id)
            else:
                rejected.append({"download_id": download.download_id, "period": link.period, "reasons": reasons})
        blockers = []
        if not usable_by_period["PRE"]:
            blockers.append("NO_USABLE_PRE")
        if not usable_by_period["POST"]:
            blockers.append("NO_USABLE_POST")
        rows.append(
            {
                "sample_id": event.sample_id,
                "event_id": event.event_id,
                "event_year": event.event_year,
                "usable_counts": {period: len(ids) for period, ids in usable_by_period.items()},
                "usable_download_ids": usable_by_period,
                "rejected_downloads": rejected,
                "blockers": blockers,
                "input_status": "READY" if not blockers else "INCOMPLETE_INPUT",
                "existing_model_sources": sorted(label_sources.get(event.event_id, set())),
            }
        )

    artifacts = {
        "REMOTECLIP_LOGREG": model_artifact_status("REMOTECLIP_LOGREG", remoteclip_version),
        "XGBOOST": model_artifact_status("XGBOOST", xgboost_version),
    }
    ready_inputs = sum(row["input_status"] == "READY" for row in rows)
    return {
        "mining_run_id": mining_run_id,
        "models": artifacts,
        "summary": {
            "samples": len(sample_ids),
            "events": len(events),
            "ready_inputs": ready_inputs,
            "incomplete_inputs": len(events) - ready_inputs,
            "remoteclip_artifact_ready": artifacts["REMOTECLIP_LOGREG"]["ready"],
            "xgboost_artifact_ready": artifacts["XGBOOST"]["ready"],
            "inference_ready": ready_inputs > 0 and any(item["ready"] for item in artifacts.values()),
        },
        "events": rows,
    }

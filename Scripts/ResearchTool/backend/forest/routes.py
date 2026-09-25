from __future__ import annotations

import csv
import json
import threading
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from ..config import FOREST_EXPORTS_DIR
from .database import SessionLocal, get_db
from .geometry import DEFAULT_SAMPLE_PLOT_M, DEFAULT_VIEWER_AOI_M, haversine_m, sample_geometries
from .hansen_service import analyze_sample_hansen, latest_hansen_payload
from .import_service import import_file
from .jobs import jobs
from .models import (
    ForestExport,
    HansenAnalysis,
    ManualValidation,
    Sample,
    SampleStatus,
    SentinelDownload,
    SentinelSceneReview,
    utc_now,
)
from .sentinel_service import (
    download_sentinel_for_sample,
    latest_sentinel_payload,
    search_sentinel_for_sample,
    sentinel_review_payload,
)
from .storage_paths import resolve_storage_path, to_storage_path


router = APIRouter(prefix="/api/forest", tags=["forest"])

MANUAL_VALIDATION_VALUES = {"Valid", "Unclear", "Wrong", "Too old"}
SENTINEL_REVIEW_REASON_CODES = {
    "CLOUDS",
    "CLOUD_SHADOW",
    "HAZE_OR_SMOKE",
    "SNOW_OR_ICE",
    "SEASON_MISMATCH",
    "TOO_DARK_OR_LOW_CONTRAST",
    "NO_DATA_OR_BLACK_PIXELS",
    "AOI_NOT_COVERED",
    "GEOREGISTRATION_SHIFT",
    "WRONG_EVENT_WINDOW",
    "PREVIEW_OR_PROCESSING_ARTIFACT",
    "OTHER",
}
VALID_FOR_NEW_SEARCH_FILTER = "__valid_for_new_search"
NO_MANUAL_VALIDATION_FILTER = "__no_manual_validation"
SPECIAL_MANUAL_VALIDATION_FILTERS = {
    VALID_FOR_NEW_SEARCH_FILTER,
    NO_MANUAL_VALIDATION_FILTER,
}
HANSEN_SAMPLE_JOB_TOTAL = 5
HANSEN_STAGE_PROGRESS = {
    "queued": 0,
    "loading": 1,
    "cache": 2,
    "analysis": 4,
    "done": 5,
    "failed": 5,
    "cancelled": 5,
}


class ManualValidationRequest(BaseModel):
    validation: str
    notes: str | None = None


class SampleNameRequest(BaseModel):
    display_name: str | None = Field(default=None, max_length=255)


class SampleFlagsRequest(BaseModel):
    has_multiple_events: bool | None = None


class SentinelSearchRequest(BaseModel):
    event_year: int | None = None
    season_preset: str = "full_snow_free"
    max_cloud: float = Field(default=30.0, ge=0.0, le=100.0)
    top_n: int = Field(default=10, ge=1, le=50)


class SentinelDownloadRequest(BaseModel):
    item: dict[str, Any]
    period: str | None = None
    max_cloud: float = Field(default=30.0, ge=0.0, le=100.0)


class SentinelReviewRequest(BaseModel):
    is_excluded: bool = False
    reason_code: str | None = Field(default=None, max_length=64)
    reason_text: str | None = None
    notes: str | None = None


class HansenAnalysisRequest(BaseModel):
    include_treecover: bool = False


class HansenBatchRequest(BaseModel):
    filters: dict[str, Any] = Field(default_factory=dict)
    limit: int = Field(default=10, ge=1, le=1000)
    include_treecover: bool = False
    skip_existing: bool = True


class ExportRequest(BaseModel):
    scope: str = "current_filtered"
    filters: dict[str, Any] = Field(default_factory=dict)


def json_or_none(value: str | None) -> Any:
    if value is None:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def dt_iso(value: Any) -> str | None:
    return value.isoformat() if value is not None else None


def sample_summary(sample: Sample, *, distance_m: float | None = None) -> dict[str, Any]:
    payload = {
        "sample_id": sample.sample_id,
        "short_id": sample.sample_id[:8],
        "display_name": sample.display_name,
        "source_file": sample.source_file,
        "source_format": sample.source_format,
        "source_id": sample.source_id,
        "lat": sample.lat,
        "lon": sample.lon,
        "source_event_year": sample.source_event_year,
        "driver_primary": sample.driver_primary,
        "driver_primary_code": sample.driver_primary_code,
        "confidence_primary": sample.confidence_primary,
        "driver_secondary": sample.driver_secondary,
        "confidence_secondary": sample.confidence_secondary,
        "region": sample.region,
        "region_code": sample.region_code,
        "tags": sample.tags,
        "notes": sample.notes,
        "has_multiple_events": bool(sample.has_multiple_events),
        "manual_validation": sample.manual_validation.validation
        if sample.manual_validation is not None
        else None,
        "manual_notes": sample.manual_validation.notes if sample.manual_validation is not None else None,
        "is_saved": sample.manual_validation is not None,
        "created_at": dt_iso(sample.created_at),
        "updated_at": dt_iso(sample.updated_at),
    }
    if distance_m is not None:
        payload["distance_m"] = round(distance_m, 2)
    return payload


def sample_detail(sample: Sample, db: Session) -> dict[str, Any]:
    payload = sample_summary(sample)
    payload["raw_properties"] = json_or_none(sample.raw_properties_json)
    payload["statuses"] = [
        {
            "source": status.source,
            "status": status.status,
            "reason": status.reason,
            "metrics": json_or_none(status.metrics_json),
            "updated_at": dt_iso(status.updated_at),
        }
        for status in sample.statuses
    ]
    payload["geometry"] = sample_geometries(
        sample.lon,
        sample.lat,
        sample_plot_m=DEFAULT_SAMPLE_PLOT_M,
        viewer_aoi_m=DEFAULT_VIEWER_AOI_M,
    )
    payload["hansen"] = latest_hansen_payload(db, sample.sample_id)
    payload["sentinel"] = latest_sentinel_payload(db, sample.sample_id)
    return payload


def sample_conditions(
    *,
    q: str | None = None,
    id_q: str | None = None,
    driver: str | None = None,
    confidence: str | None = None,
    region: str | None = None,
    manual_validation: str | None = None,
) -> list[Any]:
    conditions: list[Any] = []
    if q:
        pattern = f"%{q}%"
        conditions.append(
            or_(
                Sample.sample_id.ilike(pattern),
                Sample.display_name.ilike(pattern),
                Sample.source_id.ilike(pattern),
                Sample.source_file.ilike(pattern),
            )
        )
    if id_q:
        pattern = f"%{id_q}%"
        conditions.append(
            or_(
                Sample.sample_id.ilike(pattern),
                Sample.source_id.ilike(pattern),
            )
        )
    if driver:
        conditions.append(Sample.driver_primary == driver)
    if confidence:
        conditions.append(Sample.confidence_primary == confidence)
    if region:
        conditions.append(Sample.region == region)
    if manual_validation:
        if manual_validation == VALID_FOR_NEW_SEARCH_FILTER:
            manual_too_old = (
                select(ManualValidation.sample_id)
                .where(
                    ManualValidation.sample_id == Sample.sample_id,
                    ManualValidation.validation == "Too old",
                )
                .exists()
            )
            hansen_too_old = (
                select(SampleStatus.sample_id)
                .where(
                    SampleStatus.sample_id == Sample.sample_id,
                    SampleStatus.source == "hansen",
                    SampleStatus.status == "TOO_OLD",
                )
                .exists()
            )
            conditions.extend([~manual_too_old, ~hansen_too_old])
        elif manual_validation == NO_MANUAL_VALIDATION_FILTER:
            has_manual_validation = (
                select(ManualValidation.sample_id)
                .where(ManualValidation.sample_id == Sample.sample_id)
                .exists()
            )
            conditions.append(~has_manual_validation)
        else:
            conditions.append(ManualValidation.validation == manual_validation)
    return conditions


def samples_select(filters: dict[str, Any], *, count: bool = False) -> Any:
    manual_validation = filters.get("manual_validation")
    stmt = select(func.count()).select_from(Sample) if count else select(Sample)
    if manual_validation and manual_validation not in SPECIAL_MANUAL_VALIDATION_FILTERS:
        stmt = stmt.join(ManualValidation, ManualValidation.sample_id == Sample.sample_id)
    conditions = sample_conditions(
        q=filters.get("q"),
        id_q=filters.get("id_q"),
        driver=filters.get("driver"),
        confidence=filters.get("confidence"),
        region=filters.get("region"),
        manual_validation=manual_validation,
    )
    if conditions:
        stmt = stmt.where(*conditions)
    if not count:
        stmt = stmt.order_by(Sample.source_file, Sample.source_id, Sample.sample_id)
    return stmt


def read_filtered_samples(db: Session, filters: dict[str, Any]) -> list[Sample]:
    return list(db.scalars(samples_select(filters)))


def job_cancel_requested(job_id: str) -> bool:
    job = jobs.get(job_id)
    return bool(job and job.cancel_requested)


def run_hansen_sample_job(job_id: str, sample_id: str, include_treecover: bool = False) -> None:
    db = SessionLocal()

    def progress(stage: str, message: str) -> None:
        if job_cancel_requested(job_id):
            jobs.update(job_id, status="cancel_requested", stage=stage, message=message)
            return
        jobs.update(
            job_id,
            status="running",
            stage=stage,
            done=HANSEN_STAGE_PROGRESS.get(stage),
            message=message,
        )

    try:
        jobs.update(
            job_id,
            status="running",
            stage="loading",
            done=HANSEN_STAGE_PROGRESS["loading"],
            message="Loading sample",
        )
        sample = db.get(Sample, sample_id)
        if sample is None:
            jobs.update(
                job_id,
                status="failed",
                stage="failed",
                done=HANSEN_SAMPLE_JOB_TOTAL,
                message="Sample was not found.",
            )
            return

        analyze_sample_hansen(
            db,
            sample,
            include_treecover=include_treecover,
            progress=progress,
            should_cancel=lambda: job_cancel_requested(job_id),
        )
        if job_cancel_requested(job_id):
            jobs.update(
                job_id,
                status="cancelled",
                stage="cancelled",
                done=HANSEN_SAMPLE_JOB_TOTAL,
                message="Hansen analysis cancelled",
            )
            return
        jobs.update(
            job_id,
            status="completed",
            stage="done",
            done=HANSEN_SAMPLE_JOB_TOTAL,
            message=f"Hansen analysis ready for {sample.sample_id[:8]}",
        )
    except Exception as exc:
        db.rollback()
        if job_cancel_requested(job_id):
            jobs.update(
                job_id,
                status="cancelled",
                stage="cancelled",
                done=HANSEN_SAMPLE_JOB_TOTAL,
                message="Hansen analysis cancelled",
            )
        else:
            jobs.update(
                job_id,
                status="failed",
                stage="failed",
                done=HANSEN_SAMPLE_JOB_TOTAL,
                message=str(exc)[:500],
            )
    finally:
        db.close()


def run_hansen_batch_job(job_id: str, sample_ids: list[str], include_treecover: bool = False) -> None:
    db = SessionLocal()
    failed = 0
    try:
        jobs.update(
            job_id,
            status="running",
            stage="loading",
            total=len(sample_ids),
            message="Hansen batch started",
        )
        for index, sample_id in enumerate(sample_ids, start=1):
            job = jobs.get(job_id)
            if job and job.cancel_requested:
                jobs.update(
                    job_id,
                    status="cancelled",
                    stage="cancelled",
                    done=index - 1,
                    message=f"Cancelled after {index - 1}/{len(sample_ids)} samples",
                )
                return

            sample = db.get(Sample, sample_id)
            if sample is None:
                failed += 1
                jobs.update(
                    job_id,
                    done=index,
                    message=f"{sample_id[:8]} missing; failed {failed}",
                )
                continue

            try:
                analyze_sample_hansen(db, sample, include_treecover=include_treecover)
                jobs.update(
                    job_id,
                    done=index,
                    message=f"Analyzed {sample.sample_id[:8]} ({index}/{len(sample_ids)})",
                )
            except Exception as exc:
                failed += 1
                db.rollback()
                jobs.update(
                    job_id,
                    done=index,
                    message=f"{sample.sample_id[:8]} failed: {str(exc)[:160]}",
                )

        final_status = "completed_with_errors" if failed else "completed"
        jobs.update(
            job_id,
            status=final_status,
            stage="done",
            done=len(sample_ids),
            message=f"Done: {len(sample_ids) - failed} ok, {failed} failed",
        )
    except Exception as exc:
        db.rollback()
        jobs.update(job_id, status="failed", stage="failed", message=str(exc)[:500])
    finally:
        db.close()


@router.post("/import")
async def import_samples(
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    results = []
    for upload in files:
        filename = upload.filename or "upload.dat"
        content = await upload.read()
        if not content:
            results.append(
                {
                    "filename": filename,
                    "imported_count": 0,
                    "duplicate_count": 0,
                    "rejected_count": 1,
                    "rejected_examples": [{"row": None, "reason": "Uploaded file is empty."}],
                }
            )
            continue
        results.append(import_file(db, filename=filename, content=content))

    return {
        "files": results,
        "imported_count": sum(item["imported_count"] for item in results),
        "duplicate_count": sum(item["duplicate_count"] for item in results),
        "rejected_count": sum(item["rejected_count"] for item in results),
    }


@router.get("/samples")
def list_samples(
    db: Session = Depends(get_db),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    q: str | None = None,
    id_q: str | None = None,
    driver: str | None = None,
    confidence: str | None = None,
    region: str | None = None,
    manual_validation: str | None = None,
    selected_id: str | None = None,
    nearest_n: int = Query(25, ge=0, le=250),
) -> dict[str, Any]:
    filters = {
        "q": q,
        "id_q": id_q,
        "driver": driver,
        "confidence": confidence,
        "region": region,
        "manual_validation": manual_validation,
    }
    total = db.scalar(samples_select(filters, count=True)) or 0
    page_stmt = samples_select(filters).limit(limit).offset(offset)
    samples = list(db.scalars(page_stmt))

    selected = (
        db.scalar(samples_select(filters).where(Sample.sample_id == selected_id).limit(1))
        if selected_id
        else None
    )
    nearest: list[dict[str, Any]] = []
    if selected is not None and nearest_n > 0:
        candidates = read_filtered_samples(db, filters)
        ranked = sorted(
            (
                (
                    haversine_m(selected.lat, selected.lon, candidate.lat, candidate.lon),
                    candidate,
                )
                for candidate in candidates
                if candidate.sample_id != selected.sample_id
            ),
            key=lambda item: item[0],
        )
        nearest = [
            sample_summary(candidate, distance_m=distance_m)
            for distance_m, candidate in ranked[:nearest_n]
        ]

    return {
        "items": [sample_summary(sample) for sample in samples],
        "total": total,
        "limit": limit,
        "offset": offset,
        "selected": sample_summary(selected) if selected is not None else None,
        "nearest": nearest,
        "nearest_n": nearest_n,
    }


@router.get("/samples/{sample_id}")
def get_sample(sample_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    sample = db.get(Sample, sample_id)
    if sample is None:
        raise HTTPException(status_code=404, detail="Sample was not found.")
    return sample_detail(sample, db)


@router.post("/samples/{sample_id}/display-name")
def save_sample_display_name(
    sample_id: str,
    request: SampleNameRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    sample = db.get(Sample, sample_id)
    if sample is None:
        raise HTTPException(status_code=404, detail="Sample was not found.")

    name = (request.display_name or "").strip()
    sample.display_name = name or None
    sample.updated_at = utc_now()
    db.commit()
    db.refresh(sample)
    return sample_detail(sample, db)


@router.post("/samples/{sample_id}/flags")
def save_sample_flags(
    sample_id: str,
    request: SampleFlagsRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    sample = db.get(Sample, sample_id)
    if sample is None:
        raise HTTPException(status_code=404, detail="Sample was not found.")

    if request.has_multiple_events is not None:
        sample.has_multiple_events = bool(request.has_multiple_events)
    sample.updated_at = utc_now()
    db.commit()
    db.refresh(sample)
    return sample_detail(sample, db)


@router.post("/samples/{sample_id}/manual-validation")
def save_manual_validation(
    sample_id: str,
    request: ManualValidationRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    sample = db.get(Sample, sample_id)
    if sample is None:
        raise HTTPException(status_code=404, detail="Sample was not found.")
    if request.validation not in MANUAL_VALIDATION_VALUES:
        allowed = ", ".join(sorted(MANUAL_VALIDATION_VALUES))
        raise HTTPException(status_code=400, detail=f"validation must be one of: {allowed}")

    manual = db.get(ManualValidation, sample_id)
    if manual is None:
        manual = ManualValidation(sample_id=sample_id, validation=request.validation)
        db.add(manual)
    manual.validation = request.validation
    manual.notes = request.notes
    manual.updated_at = utc_now()

    status = db.scalar(
        select(SampleStatus).where(
            SampleStatus.sample_id == sample_id,
            SampleStatus.source == "manual",
        )
    )
    if status is None:
        status = SampleStatus(sample_id=sample_id, source="manual", status=request.validation)
        db.add(status)
    status.status = request.validation
    status.reason = request.notes
    status.metrics_json = None
    status.updated_at = utc_now()

    db.commit()
    db.refresh(sample)
    return sample_detail(sample, db)


def export_row(sample: Sample) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "display_name": sample.display_name,
        "source_file": sample.source_file,
        "source_id": sample.source_id,
        "lat": sample.lat,
        "lon": sample.lon,
        "driver": sample.driver_primary,
        "confidence": sample.confidence_primary,
        "has_multiple_events": bool(sample.has_multiple_events),
        "source_event_year": sample.source_event_year,
        "event_year": None,
        "total_loss_area_ha": None,
        "dominant_loss_area_ha": None,
        "dominant_year_share": None,
        "sample_status": None,
        "pre_scene_id": None,
        "pre_date": None,
        "post_scene_id": None,
        "post_date": None,
        "pre_local_cloud": None,
        "post_local_cloud": None,
        "manual_validation": sample.manual_validation.validation
        if sample.manual_validation is not None
        else None,
        "manual_notes": sample.manual_validation.notes if sample.manual_validation is not None else None,
        "geometry": json.dumps({"type": "Point", "coordinates": [sample.lon, sample.lat]}),
        "event_mask_metadata": None,
        "local_raster_paths": None,
    }


@router.post("/export")
def create_export(request: ExportRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    if request.scope not in {"current_filtered", "all"}:
        raise HTTPException(status_code=400, detail="scope must be current_filtered or all.")

    filters = {} if request.scope == "all" else request.filters
    samples = read_filtered_samples(db, filters)
    export_id = uuid.uuid4().hex
    export_dir = FOREST_EXPORTS_DIR / export_id
    export_dir.mkdir(parents=True, exist_ok=True)
    csv_path = export_dir / "forest_samples.csv"
    geojson_path = export_dir / "forest_samples.geojson"

    rows = [export_row(sample) for sample in samples]
    fieldnames = list(rows[0].keys()) if rows else list(export_row_empty().keys())
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    features = [
        {
            "type": "Feature",
            "properties": {key: value for key, value in row.items() if key != "geometry"},
            "geometry": {"type": "Point", "coordinates": [sample.lon, sample.lat]},
        }
        for row, sample in zip(rows, samples, strict=True)
    ]
    geojson_path.write_text(
        json.dumps(
            {"type": "FeatureCollection", "features": features},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    db.add(
        ForestExport(
            export_id=export_id,
            scope=request.scope,
            csv_path=to_storage_path(csv_path) or str(csv_path),
            geojson_path=to_storage_path(geojson_path) or str(geojson_path),
            sample_count=len(samples),
            filters_json=json.dumps(filters, ensure_ascii=False, sort_keys=True),
        )
    )
    db.commit()

    return {
        "export_id": export_id,
        "scope": request.scope,
        "sample_count": len(samples),
        "csv_url": f"/api/forest/exports/{export_id}?format=csv",
        "geojson_url": f"/api/forest/exports/{export_id}?format=geojson",
    }


def export_row_empty() -> dict[str, Any]:
    return {
        "sample_id": None,
        "display_name": None,
        "source_file": None,
        "source_id": None,
        "lat": None,
        "lon": None,
        "driver": None,
        "confidence": None,
        "has_multiple_events": None,
        "source_event_year": None,
        "event_year": None,
        "total_loss_area_ha": None,
        "dominant_loss_area_ha": None,
        "dominant_year_share": None,
        "sample_status": None,
        "pre_scene_id": None,
        "pre_date": None,
        "post_scene_id": None,
        "post_date": None,
        "pre_local_cloud": None,
        "post_local_cloud": None,
        "manual_validation": None,
        "manual_notes": None,
        "geometry": None,
        "event_mask_metadata": None,
        "local_raster_paths": None,
    }


@router.get("/exports/{export_id}")
def get_export(
    export_id: str,
    format: str = Query("csv", pattern="^(csv|geojson)$"),
    db: Session = Depends(get_db),
) -> FileResponse:
    export = db.get(ForestExport, export_id)
    if export is None:
        raise HTTPException(status_code=404, detail="Export was not found.")
    path = resolve_storage_path(export.csv_path if format == "csv" else export.geojson_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Export file is missing on disk.")
    media_type = "text/csv" if format == "csv" else "application/geo+json"
    return FileResponse(path, media_type=media_type, filename=path.name)


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job was not found.")
    return job.snapshot()


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, Any]:
    job = jobs.request_cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job was not found.")
    return job.snapshot()


@router.post("/samples/{sample_id}/hansen/analyze-job")
def start_hansen_sample_analysis(
    sample_id: str,
    request: HansenAnalysisRequest | None = None,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    sample = db.get(Sample, sample_id)
    if sample is None:
        raise HTTPException(status_code=404, detail="Sample was not found.")

    include_treecover = bool(request.include_treecover) if request else False
    job = jobs.create(
        "hansen_sample",
        total=HANSEN_SAMPLE_JOB_TOTAL,
        message=f"Queued Hansen analysis for {sample.sample_id[:8]}",
    )
    thread = threading.Thread(
        target=run_hansen_sample_job,
        args=(job.job_id, sample.sample_id, include_treecover),
        daemon=True,
    )
    thread.start()
    return job.snapshot()


@router.post("/samples/{sample_id}/hansen/analyze")
def analyze_hansen(
    sample_id: str,
    request: HansenAnalysisRequest | None = None,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    sample = db.get(Sample, sample_id)
    if sample is None:
        raise HTTPException(status_code=404, detail="Sample was not found.")
    try:
        include_treecover = bool(request.include_treecover) if request else False
        return analyze_sample_hansen(db, sample, include_treecover=include_treecover)
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/hansen/analyze-batch")
def analyze_hansen_batch(
    request: HansenBatchRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    # sample_ids = [
    #     sample.sample_id
    #     for sample in read_filtered_samples(db, request.filters)[: request.limit]
    # ]
    samples = read_filtered_samples(db, request.filters)

    if request.skip_existing:
        existing_ids = set(
            db.scalars(
                select(HansenAnalysis.sample_id).distinct()
            )
        )
        samples = [s for s in samples if s.sample_id not in existing_ids]

    sample_ids = [s.sample_id for s in samples[: request.limit]]
    if not sample_ids:
        raise HTTPException(status_code=400, detail="No samples matched batch filters.")

    job = jobs.create("hansen_batch", total=len(sample_ids), message="Queued Hansen batch")
    thread = threading.Thread(
        target=run_hansen_batch_job,
        args=(job.job_id, sample_ids, request.include_treecover),
        daemon=True,
    )
    thread.start()
    return job.snapshot()


@router.post("/samples/{sample_id}/sentinel/search")
def sentinel_search(
    sample_id: str,
    request: SentinelSearchRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    sample = db.get(Sample, sample_id)
    if sample is None:
        raise HTTPException(status_code=404, detail="Sample was not found.")
    event_year = request.event_year or sample.source_event_year
    hansen = latest_hansen_payload(db, sample_id)
    if event_year is None and hansen:
        event_year = hansen.get("event_year")
    if event_year is None:
        raise HTTPException(
            status_code=400,
            detail="No event year found. Run Hansen analysis first or enter an event year.",
        )
    try:
        return search_sentinel_for_sample(
            db,
            sample,
            event_year=int(event_year),
            season_preset=request.season_preset,
            max_cloud=request.max_cloud,
            top_n=request.top_n,
        )
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/samples/{sample_id}/sentinel/download")
def sentinel_download(
    sample_id: str,
    request: SentinelDownloadRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    sample = db.get(Sample, sample_id)
    if sample is None:
        raise HTTPException(status_code=404, detail="Sample was not found.")
    try:
        return download_sentinel_for_sample(
            db,
            sample,
            item=request.item,
            period=request.period,
            max_cloud=request.max_cloud,
        )
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=502, detail=str(exc)) from exc


def clean_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


@router.get("/sentinel/downloads/{download_id}/review")
def get_sentinel_review(download_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    download = db.get(SentinelDownload, download_id)
    if download is None:
        raise HTTPException(status_code=404, detail="Sentinel download was not found.")

    review = db.scalar(
        select(SentinelSceneReview).where(SentinelSceneReview.download_id == download_id)
    )
    return {
        "download_id": download_id,
        "review": sentinel_review_payload(review),
    }


@router.put("/sentinel/downloads/{download_id}/review")
def save_sentinel_review(
    download_id: str,
    request: SentinelReviewRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    download = db.get(SentinelDownload, download_id)
    if download is None:
        raise HTTPException(status_code=404, detail="Sentinel download was not found.")

    reason_code = clean_optional_text(request.reason_code)
    reason_text = clean_optional_text(request.reason_text)
    notes = clean_optional_text(request.notes)

    if request.is_excluded:
        if not reason_code:
            raise HTTPException(status_code=400, detail="reason_code is required when excluding a Sentinel scene.")
        if reason_code not in SENTINEL_REVIEW_REASON_CODES:
            allowed = ", ".join(sorted(SENTINEL_REVIEW_REASON_CODES))
            raise HTTPException(status_code=400, detail=f"reason_code must be one of: {allowed}")
        if reason_code == "OTHER" and not reason_text:
            raise HTTPException(status_code=400, detail="reason_text is required when reason_code is OTHER.")
    if not request.is_excluded:
        reason_code = None
        reason_text = None
        notes = None

    review = db.scalar(
        select(SentinelSceneReview).where(SentinelSceneReview.download_id == download_id)
    )
    if review is None:
        review = SentinelSceneReview(download_id=download_id)
        db.add(review)

    review.is_excluded = request.is_excluded
    review.reason_code = reason_code
    review.reason_text = reason_text
    review.notes = notes
    review.updated_at = utc_now()

    db.commit()
    db.refresh(review)
    return {
        "download_id": download_id,
        "review": sentinel_review_payload(review),
    }

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOL_ROOT = PROJECT_ROOT / "Scripts" / "ResearchTool"
print(TOOL_ROOT)
sys.path.insert(0, str(TOOL_ROOT))

from backend.forest.database import init_forest_database, SessionLocal
from backend.forest.models import Sample, HansenAnalysis, ManualValidation, SampleStatus, utc_now
from backend.forest.hansen_service import analyze_sample_hansen
from sqlalchemy import select

def iter_hansen_ready_unvalidated_samples(db):
    stmt = (
        select(Sample)
        .join(HansenAnalysis, HansenAnalysis.sample_id == Sample.sample_id)
        .outerjoin(ManualValidation, ManualValidation.sample_id == Sample.sample_id)
        .where(ManualValidation.sample_id.is_(None))
        .distinct()
        .order_by(Sample.source_file, Sample.source_id, Sample.sample_id)
    )
    return db.scalars(stmt)

def iter_unvalidated_samples(db):
    stmt = (
        select(Sample)
        .outerjoin(ManualValidation, ManualValidation.sample_id == Sample.sample_id)
        .where(ManualValidation.sample_id.is_(None))
        .order_by(Sample.source_file, Sample.source_id, Sample.sample_id)
    )
    return db.scalars(stmt)

def iter_hansen_too_old_samples(db):
    stmt = (
        select(Sample)
        .join(SampleStatus, SampleStatus.sample_id == Sample.sample_id)
        .where(SampleStatus.source == "hansen")
        .where(SampleStatus.status == "TOO_OLD")
        .order_by(Sample.source_file, Sample.source_id, Sample.sample_id)
    )
    return db.scalars(stmt)

def upsert_manual_validation(db, sample_id: str, validation: str, notes: str | None = None):
    manual = db.get(ManualValidation, sample_id)

    if manual is None:
        manual = ManualValidation(sample_id=sample_id, validation=validation)
        db.add(manual)

    manual.validation = validation
    manual.notes = notes
    manual.updated_at = utc_now()

    status = db.scalar(
        select(SampleStatus).where(
            SampleStatus.sample_id == sample_id,
            SampleStatus.source == "manual",
        )
    )

    if status is None:
        status = SampleStatus(sample_id=sample_id, source="manual", status=validation)
        db.add(status)

    status.status = validation
    status.reason = notes
    status.metrics_json = None
    status.updated_at = utc_now()

init_forest_database()

with SessionLocal() as db:
    for sample in iter_unvalidated_samples(db):
        print(sample.sample_id, sample.lat, sample.lon)
        payload = analyze_sample_hansen(
            db,
            sample,
            include_treecover=False,
        )

        year = payload["event_year"]
        print(year)

NOTES = "Marked by script: Hansen status TOO_OLD"

with SessionLocal() as db:
    count = 0

    for sample in iter_hansen_too_old_samples(db):
        if sample.manual_validation is not None:
            continue

        upsert_manual_validation(
            db,
            sample.sample_id,
            "Too old",
            NOTES,
        )
        count += 1

    db.commit()

print(f"Marked Too old: {count}")
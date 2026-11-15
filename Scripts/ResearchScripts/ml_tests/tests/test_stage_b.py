from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from ml_tests.stage_b import STAGE_B_TABLES, _schema_report

from ResearchTool.backend.forest.event_review_service import save_event_review
from ResearchTool.backend.forest.models import (
    Base,
    CandidatePreparationRun,
    EventReview,
    EventReviewReason,
    EventReviewRevision,
    HansenCandidate,
    HansenCandidateRun,
    HansenCandidateRunItem,
    PreparedEvent,
    PreparedEventMember,
    Sample,
    SampleEvent,
)
from ResearchTool.backend.forest.pipeline_v2_migration import (
    apply_event_review_reason_migration,
    apply_legacy_prepared_event_backfill,
    assert_legacy_backfill_plan_safe,
    assert_review_migration_plan_safe,
    plan_event_review_reason_migration,
    plan_legacy_prepared_event_backfill,
)
from ResearchTool.backend.forest.review_reason_service import map_legacy_rejection_reason


def sample(sample_id: str) -> Sample:
    return Sample(
        sample_id=sample_id,
        source_file="test",
        source_format="HANSEN_MINING",
        source_id=sample_id,
        lat=55.0,
        lon=37.0,
        driver_primary="Logging",
        raw_properties_json="{}",
    )


def event(event_id: str, sample_id: str, year: int) -> SampleEvent:
    return SampleEvent(
        event_id=event_id,
        sample_id=sample_id,
        event_year=year,
        source="HANSEN",
        loss_area_ha=2.0,
        loss_share=0.75,
        is_dominant=True,
        metadata_json="{}",
    )


def candidate_run(run_id: str) -> HansenCandidateRun:
    return HansenCandidateRun(
        run_id=run_id,
        algorithm_name="test",
        algorithm_version="v1",
        hansen_dataset_version="v1.12",
        status="COMPLETED",
        target_count=1,
        selected_count=1,
        created_sample_count=1,
        config_json="{}",
    )


class StageBTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "stage_b.sqlite"
        self.engine = create_engine(f"sqlite:///{self.db_path.as_posix()}", future=True)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine, future=True)

    def tearDown(self) -> None:
        self.engine.dispose()
        self.temp_dir.cleanup()

    def test_schema_contains_every_stage_b_table(self) -> None:
        report = _schema_report(self.db_path)
        self.assertFalse(report["missing_stage_b_tables"])
        self.assertTrue(STAGE_B_TABLES.issubset(set(report["after_tables"])))
        self.assertFalse(report["changed_legacy_counts"])

    def test_reason_mapping_is_multilingual_and_does_not_hide_unknowns(self) -> None:
        self.assertEqual(
            map_legacy_rejection_reason("A small event for recognizing the type.").reason_codes,
            ("EVENT_TOO_SMALL",),
        )
        self.assertEqual(
            map_legacy_rejection_reason("No valid PRE-images").reason_codes,
            ("QC_FAILED",),
        )
        self.assertEqual(
            map_legacy_rejection_reason("The consequences of the fire next year.").reason_codes,
            ("YEAR_MISMATCH",),
        )
        self.assertEqual(
            map_legacy_rejection_reason(
                "Между изоюражениями не видно рвзницы"
            ).reason_codes,
            ("NO_VISIBLE_CHANGE",),
        )
        mixed = map_legacy_rejection_reason(
            "Явно не пожар, но может и не вырубка - слишком маленькое событие"
        )
        self.assertEqual(mixed.reason_codes, ("EVENT_TOO_SMALL", "OTHER_INVALID"))
        self.assertEqual(map_legacy_rejection_reason("unknown custom text").mapping_status, "UNMAPPED")

    def test_review_migration_and_future_save_are_append_only(self) -> None:
        with self.SessionLocal() as db:
            db.add_all([sample("s1"), sample("s2")])
            db.add_all([event("e1", "s1", 2020), event("e2", "s2", 2021)])
            db.flush()
            db.add_all(
                [
                    EventReview(
                        event_id="e1",
                        review_status="REJECTED",
                        rejection_reason="A small event for recognizing the type.",
                        reviewer_name="legacy",
                    ),
                    EventReview(event_id="e2", review_status="DRAFT"),
                ]
            )
            db.commit()

            plan = plan_event_review_reason_migration(db)
            assert_review_migration_plan_safe(plan)
            first = apply_event_review_reason_migration(db, plan)
            second = apply_event_review_reason_migration(db)
            self.assertEqual(first, {"inserted_reason_rows": 1, "inserted_revision_rows": 2})
            self.assertEqual(second, {"inserted_reason_rows": 0, "inserted_revision_rows": 0})
            self.assertEqual(db.get(EventReview, "e1").rejection_reason, "A small event for recognizing the type.")
            self.assertEqual(
                list(db.scalars(select(EventReviewReason.reason_code))),
                ["EVENT_TOO_SMALL"],
            )

            save_event_review(
                db,
                "e1",
                review_status="REJECTED",
                actor_name="reviewer",
                reason_codes=["YEAR_MISMATCH", "EVENT_TOO_SMALL"],
                rejection_reason="two independent causes",
            )
            current_codes = set(
                db.scalars(
                    select(EventReviewReason.reason_code).where(EventReviewReason.event_id == "e1")
                )
            )
            self.assertEqual(current_codes, {"YEAR_MISMATCH", "EVENT_TOO_SMALL"})
            revisions = list(
                db.scalars(
                    select(EventReviewRevision).where(EventReviewRevision.event_id == "e1")
                )
            )
            self.assertEqual(len(revisions), 2)
            self.assertEqual(json.loads(revisions[-1].reason_codes_json), ["EVENT_TOO_SMALL", "YEAR_MISMATCH"])

            third = apply_event_review_reason_migration(db)
            self.assertEqual(third, {"inserted_reason_rows": 0, "inserted_revision_rows": 0})
            legacy_revisions = list(
                db.scalars(
                    select(EventReviewRevision).where(
                        EventReviewRevision.event_id == "e1",
                        EventReviewRevision.revision_source == "LEGACY_MIGRATION",
                    )
                )
            )
            self.assertEqual(len(legacy_revisions), 1)

    def test_legacy_backfill_is_run_scoped_and_idempotent(self) -> None:
        with self.SessionLocal() as db:
            db.add_all([sample("s1"), sample("s2")])
            db.add_all([event("e1", "s1", 2020), event("e2", "s2", 2021)])
            db.add_all([candidate_run("r1"), candidate_run("r2")])
            db.add(
                HansenCandidate(
                    candidate_id="c1",
                    algorithm_version="v1",
                    hansen_dataset_version="v1.12",
                    lat=55.0,
                    lon=37.0,
                    aoi_size_m=2000,
                    aoi_bbox_json="[]",
                    primary_event_year=2020,
                    event_years_json="[2020]",
                    has_multiple_events=False,
                    component_area_ha=2.0,
                    dominant_loss_area_ha=2.0,
                    dominant_loss_fraction=1.0,
                    total_loss_area_ha=2.0,
                    post_2016_loss_aoi_percent=1.0,
                    dominant_event_aoi_percent=1.0,
                    pixel_count=20,
                    source_tile_ids_json="[]",
                    component_key="test",
                    created_sample_id="s1",
                )
            )
            db.add_all(
                [
                    HansenCandidateRunItem(
                        id="i1",
                        run_id="r1",
                        candidate_id="c1",
                        decision="SELECTED",
                    ),
                    HansenCandidateRunItem(
                        id="i2",
                        run_id="r2",
                        candidate_id="c1",
                        decision="SELECTED",
                    ),
                ]
            )
            db.commit()

            plan = plan_legacy_prepared_event_backfill(db, database_fingerprint="db-sha")
            assert_legacy_backfill_plan_safe(plan)
            self.assertEqual(plan["summary"]["source_runs"], 3)
            self.assertEqual(plan["summary"]["source_events"], 2)
            self.assertEqual(plan["summary"]["context_event_rows"], 3)
            self.assertEqual(plan["summary"]["multi_context_rows"], 1)
            first = apply_legacy_prepared_event_backfill(db, plan)
            second_plan = plan_legacy_prepared_event_backfill(db, database_fingerprint="db-sha")
            second = apply_legacy_prepared_event_backfill(db, second_plan)
            self.assertEqual(
                first,
                {
                    "created_preparation_runs": 3,
                    "created_prepared_events": 3,
                    "created_prepared_event_members": 3,
                },
            )
            self.assertEqual(
                second,
                {
                    "created_preparation_runs": 0,
                    "created_prepared_events": 0,
                    "created_prepared_event_members": 0,
                },
            )
            self.assertEqual(len(list(db.scalars(select(CandidatePreparationRun)))), 3)
            self.assertEqual(len(list(db.scalars(select(PreparedEvent)))), 3)
            self.assertEqual(len(list(db.scalars(select(PreparedEventMember)))), 3)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_bounds
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session


TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from backend.forest.event_service import (  # noqa: E402
    backfill_hansen_events,
    event_id_for,
    upsert_event_download_link,
)
from backend.forest.event_review_service import save_event_review  # noqa: E402
from backend.forest.hansen_service import hansen_tile_preview_payload  # noqa: E402
from backend.forest.db_sync_service import apply_operation, compare_portable_tables  # noqa: E402
from backend.forest.model_labeling_service import build_model_labeling_plan  # noqa: E402
from backend.forest.model_inference_service import (  # noqa: E402
    InferenceEvent,
    ModelInputIssue,
    Prediction,
    XGBoostRunner,
    execute_model_labeling,
    file_sha256,
    stable_hash,
)
from backend.forest.sentinel_service import cloud_metrics  # noqa: E402
from backend.forest.candidate_service import (  # noqa: E402
    ALGORITHM_NAME,
    ALGORITHM_VERSION,
    CandidateMiningOptions,
    candidate_mask_run_status,
    candidate_hansen_preview_payload,
    generate_candidate_masks_for_run,
    hansen_tile_inventory,
    materialize_candidates,
    mine_hansen_candidates,
    stable_id,
)
from backend.forest.models import (  # noqa: E402
    Base,
    DerivedPreview,
    EventLabel,
    EventReview,
    HansenCandidate,
    HansenCandidateRun,
    HansenCandidateRunItem,
    HansenAnalysis,
    HansenMask,
    HansenTile,
    LabelRun,
    ManualValidation,
    Sample,
    SampleAssignment,
    SampleEvent,
    SampleEventDownload,
    SentinelDownload,
)
from backend.forest.routes import SentinelBatchRequest, samples_select, selected_batch_samples  # noqa: E402
from backend.forest.sentinel_batch_service import (  # noqa: E402
    BatchOptions,
    execute_sentinel_batch,
    period_season_qc_caps,
    plan_sentinel_batch,
    season_qc_caps,
    season_valid_targets,
)


def sample(sample_id: str = "sample-1") -> Sample:
    return Sample(
        sample_id=sample_id,
        source_file="fixture.csv",
        source_format="csv",
        source_id=sample_id,
        display_name=sample_id,
        lat=60.0,
        lon=30.0,
        driver_primary="Logging",
        raw_properties_json="{}",
    )


def analysis(sample_id: str, analysis_id: str, rows: list[dict], dominant: int) -> HansenAnalysis:
    return HansenAnalysis(
        analysis_id=analysis_id,
        sample_id=sample_id,
        analysis_area_type="plot",
        plot_size_m=500.0,
        event_year=dominant,
        total_loss_area_ha=sum(float(row["loss_area_ha"]) for row in rows),
        dominant_loss_area_ha=max(float(row["loss_area_ha"]) for row in rows),
        dominant_year_share=0.5,
        status="READY",
        histogram_json=json.dumps(rows),
    )


class DataOperationsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
        Base.metadata.create_all(self.engine)
        self.db = Session(self.engine, autoflush=False)

    def tearDown(self) -> None:
        self.db.close()
        self.engine.dispose()

    def test_sentinel_qc_marks_scl_snow_as_bad_quality(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tif_path = Path(temp_dir) / "snow.tif"
            data = np.zeros((8, 4, 4), dtype=np.uint16)
            data[:6] = 1200
            data[6] = 11
            data[7] = 1
            with rasterio.open(
                tif_path,
                "w",
                driver="GTiff",
                width=4,
                height=4,
                count=8,
                dtype="uint16",
                transform=from_bounds(30.0, 59.0, 31.0, 60.0, 4, 4),
            ) as target:
                target.write(data)

            metrics = cloud_metrics(tif_path)

        self.assertEqual(metrics["snow_fraction"], 1.0)
        self.assertTrue(metrics["is_bad_snow"])
        self.assertTrue(metrics["is_bad_quality"])
        self.assertIn("SNOW_OR_ICE", metrics["quality_flags"])

    def test_xgboost_rejects_event_without_complete_spectral_inputs(self) -> None:
        item = sample("spectral-missing")
        event = SampleEvent(
            event_id=event_id_for(item.sample_id, "HANSEN", 2020),
            sample_id=item.sample_id,
            event_year=2020,
            source="HANSEN",
            is_dominant=True,
        )
        pre = SentinelDownload(
            download_id="spectral-pre",
            sample_id=item.sample_id,
            scene_id="pre",
            period="PRE",
            local_path="pre.tif",
            bbox_json="[]",
            bands_json="[]",
            metadata_json="{}",
        )
        post = SentinelDownload(
            download_id="spectral-post",
            sample_id=item.sample_id,
            scene_id="post",
            period="POST",
            local_path="post.tif",
            bbox_json="[]",
            bands_json="[]",
            metadata_json="{}",
        )
        runner = XGBoostRunner.__new__(XGBoostRunner)
        runner._stats_cache = {}
        inference_item = InferenceEvent(
            event=event,
            sample=item,
            downloads_by_period={"PRE": [pre], "POST": [post]},
            previews={},
            sample_events=[event],
        )

        with patch.object(runner, "_spectral_stats", return_value={}):
            with self.assertRaises(ModelInputIssue) as raised:
                runner.predict(inference_item)

        self.assertEqual(raised.exception.code, "NO_SPECTRAL_BOTH")

    def test_hansen_backfill_is_deterministic_and_plannable(self) -> None:
        item = sample()
        histogram = [
            {"year": 2014, "loss_area_ha": 9.0, "pixels": 90},
            {"year": 2016, "loss_area_ha": 1.0, "pixels": 10},
            {"year": 2018, "loss_area_ha": 2.0, "pixels": 20},
            {"year": 2020, "loss_area_ha": 3.0, "pixels": 30},
        ]
        self.db.add_all([item, analysis(item.sample_id, "analysis-1", histogram, 2020)])
        self.db.commit()

        dry_run = backfill_hansen_events(self.db, dry_run=True)
        self.assertEqual(dry_run["event_rows"], 3)
        self.assertEqual(self.db.scalar(select(SampleEvent).limit(1)), None)

        first = backfill_hansen_events(self.db, dry_run=False)
        second = backfill_hansen_events(self.db, dry_run=False)
        events = list(self.db.scalars(select(SampleEvent).order_by(SampleEvent.event_year)))
        self.db.refresh(item)

        self.assertEqual(first["event_rows"], 3)
        self.assertEqual(second["event_rows"], 3)
        self.assertEqual([event.event_year for event in events], [2016, 2018, 2020])
        self.assertEqual(
            events[0].event_id,
            event_id_for(item.sample_id, "HANSEN", 2016),
        )
        self.assertTrue(item.has_multiple_events)

        plan = plan_sentinel_batch(
            self.db,
            [item],
            BatchOptions(
                mode="empty_samples",
                seasons=("spring",),
                target_valid=3,
                max_qc_scenes=6,
            ),
        )
        self.assertEqual(plan["summary"]["samples_actionable"], 1)
        self.assertEqual(plan["summary"]["operation_rows"], 6)
        self.assertEqual(plan["samples"][0]["target_valid"], 18)
        self.assertEqual(plan["samples"][0]["max_qc_scenes"], 18)

    def test_sentinel_batch_balances_valid_and_qc_budgets_by_season(self) -> None:
        options = BatchOptions(
            mode="multi_event_enrichment",
            seasons=("spring", "early_summer", "late_summer", "autumn"),
            target_valid=10,
            max_qc_scenes=50,
        )

        self.assertEqual(
            season_valid_targets(options),
            {"spring": 2, "early_summer": 3, "late_summer": 3, "autumn": 2},
        )
        self.assertEqual(
            season_qc_caps(options),
            {"spring": 10, "early_summer": 15, "late_summer": 15, "autumn": 10},
        )
        self.assertEqual(
            period_season_qc_caps(options),
            {
                ("PRE", "spring"): 5,
                ("POST", "spring"): 5,
                ("PRE", "early_summer"): 8,
                ("POST", "early_summer"): 7,
                ("PRE", "late_summer"): 7,
                ("POST", "late_summer"): 8,
                ("PRE", "autumn"): 5,
                ("POST", "autumn"): 5,
            },
        )

    def test_shared_scene_fills_overlapping_event_period_targets(self) -> None:
        item = sample("sample-overlap-quota")
        self.db.add(item)
        self.db.add(
            analysis(
                item.sample_id,
                "analysis-overlap-quota",
                [
                    {"year": 2018, "loss_area_ha": 2.0, "pixels": 20},
                    {"year": 2020, "loss_area_ha": 3.0, "pixels": 30},
                ],
                2020,
            )
        )
        self.db.commit()
        backfill_hansen_events(self.db, dry_run=False)
        self.db.add(
            SentinelDownload(
                download_id="download-overlap-quota",
                sample_id=item.sample_id,
                scene_id="scene-overlap-quota",
                period="POST",
                local_path="cache/overlap-quota.tif",
                bbox_json="[29.9, 59.9, 30.1, 60.1]",
                bands_json="[]",
                cloud_fraction=0.0,
                shadow_fraction=0.0,
                nodata_fraction=0.0,
                dark_fraction=0.0,
                is_bad_cloud=False,
                is_bad_quality=False,
                quality_flags_json="[]",
                metadata_json=json.dumps(
                    {"item": {"id": "scene-overlap-quota", "datetime": "2019-06-20T10:00:00Z"}}
                ),
            )
        )
        self.db.commit()

        plan = plan_sentinel_batch(
            self.db,
            [item],
            BatchOptions(mode="multi_event_enrichment"),
        )
        sample_plan = plan["samples"][0]
        filled = {
            (row["event_year"], row["period"], row["season"]): row["valid_count"]
            for row in sample_plan["operations"]
            if row["valid_count"]
        }

        self.assertEqual(sample_plan["valid_count"], 2)
        self.assertEqual(filled[(2018, "POST", "early_summer")], 1)
        self.assertEqual(filled[(2020, "PRE", "early_summer")], 1)
        self.assertNotIn((2018, "PRE", "early_summer"), filled)

    def test_batch_executor_fills_each_period_and_season_target(self) -> None:
        item = sample("sample-balanced-executor")
        event = SampleEvent(
            event_id="event-balanced-executor",
            sample_id=item.sample_id,
            event_year=2020,
            source="HANSEN",
            is_dominant=True,
        )
        self.db.add_all([item, event])
        self.db.commit()

        def fake_search(*, period, **_kwargs):
            month_day = "05-10" if period.start_mmdd == "04-15" else "06-20"
            scene_id = f"scene-{period.year}-{period.start_mmdd}"
            return {
                "request": {"year": period.year, "start": period.start_mmdd},
                "items": [
                    {
                        "id": scene_id,
                        "datetime": f"{period.year}-{month_day}T10:00:00Z",
                        "cloud_cover": 0.0,
                    }
                ],
            }

        def fake_download(db, selected_sample, *, item, period, **_kwargs):
            download_id = f"download-{item['id']}"
            download = SentinelDownload(
                download_id=download_id,
                sample_id=selected_sample.sample_id,
                scene_id=item["id"],
                period=period,
                local_path=f"cache/{download_id}.tif",
                bbox_json="[29.9, 59.9, 30.1, 60.1]",
                bands_json="[]",
                cloud_fraction=0.0,
                shadow_fraction=0.0,
                nodata_fraction=0.0,
                dark_fraction=0.0,
                is_bad_cloud=False,
                is_bad_quality=False,
                quality_flags_json="[]",
                metadata_json=json.dumps({"item": item}),
            )
            db.add(download)
            db.flush()
            return {"download_id": download_id}

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = Path(temp_dir) / "sentinel"
            artifact_dir.mkdir()
            with (
                patch(
                    "backend.forest.sentinel_batch_service.stac_search_s2",
                    side_effect=fake_search,
                ),
                patch(
                    "backend.forest.sentinel_batch_service.download_sentinel_for_sample",
                    side_effect=fake_download,
                ),
                patch(
                    "backend.forest.sentinel_batch_service.output_dir",
                    return_value=artifact_dir,
                ),
            ):
                result = execute_sentinel_batch(
                    self.db,
                    [item],
                    BatchOptions(
                        mode="multi_event_enrichment",
                        seasons=("spring", "early_summer"),
                        target_valid=2,
                        max_qc_scenes=4,
                    ),
                )

        links = list(self.db.scalars(select(SampleEventDownload)))
        self.assertEqual(result["downloads_created"], 4)
        self.assertEqual(result["actions"].get("downloaded_valid"), 4)
        self.assertEqual(result["actions"].get("sample_done"), 1)
        self.assertEqual(Counter(link.period for link in links), {"PRE": 2, "POST": 2})

    def test_batch_target_counts_only_qc_valid_scenes(self) -> None:
        item = sample("sample-qc-valid-target")
        event = SampleEvent(
            event_id="event-qc-valid-target",
            sample_id=item.sample_id,
            event_year=2020,
            source="HANSEN",
            is_dominant=True,
        )
        self.db.add_all([item, event])
        self.db.commit()

        def fake_search(*, period, **_kwargs):
            return {
                "request": {"year": period.year},
                "items": [
                    {
                        "id": f"bad-{period.year}",
                        "datetime": f"{period.year}-05-08T10:00:00Z",
                        "cloud_cover": 0.0,
                    },
                    {
                        "id": f"good-{period.year}",
                        "datetime": f"{period.year}-05-20T10:00:00Z",
                        "cloud_cover": 0.0,
                    },
                ],
            }

        def fake_download(db, selected_sample, *, item, period, **_kwargs):
            download_id = f"download-{item['id']}"
            is_bad = item["id"].startswith("bad-")
            download = SentinelDownload(
                download_id=download_id,
                sample_id=selected_sample.sample_id,
                scene_id=item["id"],
                period=period,
                local_path=f"cache/{download_id}.tif",
                bbox_json="[29.9, 59.9, 30.1, 60.1]",
                bands_json="[]",
                cloud_fraction=0.0,
                shadow_fraction=0.0,
                nodata_fraction=0.5 if is_bad else 0.0,
                dark_fraction=0.0,
                is_bad_cloud=False,
                is_bad_quality=is_bad,
                quality_flags_json='["NO_DATA_OR_BLACK_PIXELS"]' if is_bad else "[]",
                metadata_json=json.dumps({"item": item}),
            )
            db.add(download)
            db.flush()
            return {"download_id": download_id}

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = Path(temp_dir) / "sentinel"
            artifact_dir.mkdir()
            with (
                patch(
                    "backend.forest.sentinel_batch_service.stac_search_s2",
                    side_effect=fake_search,
                ),
                patch(
                    "backend.forest.sentinel_batch_service.download_sentinel_for_sample",
                    side_effect=fake_download,
                ),
                patch(
                    "backend.forest.sentinel_batch_service.output_dir",
                    return_value=artifact_dir,
                ),
            ):
                result = execute_sentinel_batch(
                    self.db,
                    [item],
                    BatchOptions(
                        mode="multi_event_enrichment",
                        seasons=("spring",),
                        target_valid=1,
                        max_qc_scenes=4,
                    ),
                )

        self.assertEqual(result["downloads_created"], 4)
        self.assertEqual(result["actions"].get("downloaded_excluded"), 2)
        self.assertEqual(result["actions"].get("downloaded_valid"), 2)
        self.assertEqual(result["actions"].get("sample_done"), 1)

    def test_single_event_backfill_links_legacy_download(self) -> None:
        item = sample("sample-2")
        self.db.add(item)
        self.db.add(
            analysis(
                item.sample_id,
                "analysis-2",
                [{"year": 2019, "loss_area_ha": 4.0, "pixels": 40}],
                2019,
            )
        )
        self.db.add(
            SentinelDownload(
                download_id="download-1",
                sample_id=item.sample_id,
                scene_id="scene-1",
                period="PRE",
                local_path="cache/example.tif",
                bbox_json="[29.9, 59.9, 30.1, 60.1]",
                bands_json="[]",
                metadata_json="{}",
            )
        )
        self.db.commit()

        backfill_hansen_events(self.db, dry_run=False)
        links = list(self.db.scalars(select(SampleEventDownload)))

        self.assertEqual(len(links), 1)
        self.assertEqual(links[0].download_id, "download-1")
        self.assertEqual(links[0].period, "PRE")

    def test_one_download_can_have_different_roles_for_two_events(self) -> None:
        item = sample("sample-3")
        self.db.add(item)
        self.db.add(
            analysis(
                item.sample_id,
                "analysis-3",
                [
                    {"year": 2018, "loss_area_ha": 2.0, "pixels": 20},
                    {"year": 2020, "loss_area_ha": 3.0, "pixels": 30},
                ],
                2020,
            )
        )
        download = SentinelDownload(
            download_id="download-shared",
            sample_id=item.sample_id,
            scene_id="scene-shared",
            period="POST",
            local_path="cache/shared.tif",
            bbox_json="[29.9, 59.9, 30.1, 60.1]",
            bands_json="[]",
            metadata_json="{}",
        )
        self.db.add(download)
        self.db.commit()
        backfill_hansen_events(self.db, dry_run=False)

        events = list(self.db.scalars(select(SampleEvent).order_by(SampleEvent.event_year)))
        upsert_event_download_link(
            self.db,
            event_id=events[0].event_id,
            download_id=download.download_id,
            period="POST",
        )
        upsert_event_download_link(
            self.db,
            event_id=events[1].event_id,
            download_id=download.download_id,
            period="PRE",
        )
        self.db.commit()

        links = list(
            self.db.scalars(
                select(SampleEventDownload).order_by(SampleEventDownload.period)
            )
        )
        self.db.refresh(download)
        self.assertEqual(len(links), 2)
        self.assertEqual({link.period for link in links}, {"PRE", "POST"})
        self.assertEqual(download.period, "POST")

    def test_sample_filter_can_require_or_exclude_sentinel_downloads(self) -> None:
        with_download = sample("sample-with-download")
        without_download = sample("sample-without-download")
        self.db.add_all([with_download, without_download])
        self.db.add(
            SentinelDownload(
                download_id="download-for-filter",
                sample_id=with_download.sample_id,
                scene_id="scene-for-filter",
                period="POST",
                local_path="cache/filter.tif",
                bbox_json="[29.9, 59.9, 30.1, 60.1]",
                bands_json="[]",
                metadata_json="{}",
            )
        )
        self.db.commit()

        with_rows = list(
            self.db.scalars(samples_select({"sentinel_availability": "has_downloads"}))
        )
        without_rows = list(
            self.db.scalars(samples_select({"sentinel_availability": "no_downloads"}))
        )

        self.assertEqual([row.sample_id for row in with_rows], [with_download.sample_id])
        self.assertEqual([row.sample_id for row in without_rows], [without_download.sample_id])

    def test_sentinel_batch_can_select_samples_by_mining_run(self) -> None:
        mined = sample("mined-sample")
        mined.driver_primary = None
        other = sample("other-sample")
        self.db.add_all([mined, other])
        self.db.add(
            SampleAssignment(
                sample_id=mined.sample_id,
                assignment_type="hansen_mining_run",
                assignment_name="run-download",
            )
        )
        self.db.commit()

        rows = selected_batch_samples(
            self.db,
            SentinelBatchRequest(
                mode="hansen_mining_run",
                mining_run_id="run-download",
                drivers=["Logging", "Wildfire"],
            ),
        )

        self.assertEqual([row.sample_id for row in rows], [mined.sample_id])

    def test_event_review_rollup_keeps_partial_valid_sample(self) -> None:
        item = sample("sample-event-review")
        item.source_format = "HANSEN_MINING"
        item.driver_primary = None
        first = SampleEvent(
            event_id=event_id_for(item.sample_id, "HANSEN", 2019),
            sample_id=item.sample_id,
            event_year=2019,
            source="HANSEN",
        )
        second = SampleEvent(
            event_id=event_id_for(item.sample_id, "HANSEN", 2021),
            sample_id=item.sample_id,
            event_year=2021,
            source="HANSEN",
        )
        self.db.add_all([item, first, second])
        self.db.add_all(
            [
                EventReview(event_id=first.event_id, review_status="DRAFT"),
                EventReview(event_id=second.event_id, review_status="DRAFT"),
            ]
        )
        self.db.commit()

        result = save_event_review(
            self.db,
            first.event_id,
            review_status="VALID",
            label="Logging",
            actor_name="tester",
        )

        manual = self.db.get(ManualValidation, item.sample_id)
        self.assertEqual(result["sample_rollup"]["completion"], "PARTIAL")
        self.assertEqual(manual.validation, "Valid")
        self.assertEqual(self.db.get(EventReview, second.event_id).review_status, "DRAFT")

    def test_materialized_candidate_has_draft_events_and_no_manual_validation(self) -> None:
        run = HansenCandidateRun(
            run_id="run-1",
            algorithm_name=ALGORITHM_NAME,
            algorithm_version=ALGORITHM_VERSION,
            hansen_dataset_version="GFC-test",
            status="PLANNED",
            target_count=1,
            selected_count=1,
            config_json="{}",
        )
        candidate = HansenCandidate(
            candidate_id="candidate-1",
            algorithm_version=ALGORITHM_VERSION,
            hansen_dataset_version="GFC-test",
            lat=60.0,
            lon=30.0,
            aoi_size_m=2000.0,
            aoi_bbox_json="[29.9, 59.9, 30.1, 60.1]",
            primary_event_year=2020,
            event_years_json="[2018, 2020]",
            has_multiple_events=False,
            component_area_ha=20.0,
            dominant_loss_area_ha=10.0,
            dominant_loss_fraction=0.625,
            total_loss_area_ha=16.0,
            post_2016_loss_aoi_percent=4.0,
            dominant_event_aoi_percent=2.5,
            pixel_count=180,
            source_tile_ids_json='["60N_030E"]',
            component_key="60N_030E:2020:1",
            metadata_json=json.dumps(
                {
                    "histogram": [
                        {"year": 2018, "pixels": 60, "loss_area_ha": 6.0},
                        {"year": 2020, "pixels": 120, "loss_area_ha": 10.0},
                    ]
                }
            ),
        )
        run_item = HansenCandidateRunItem(
            id=stable_id("candidate-run-item", run.run_id, candidate.candidate_id),
            run_id=run.run_id,
            candidate_id=candidate.candidate_id,
            decision="SELECTED",
            rank=1,
            candidate_score=30.0,
        )
        self.db.add_all([run, candidate, run_item])
        self.db.commit()

        first = materialize_candidates(self.db, run.run_id, [candidate.candidate_id])
        second = materialize_candidates(self.db, run.run_id, [candidate.candidate_id])
        created_sample_id = first["sample_ids"][0]
        events = list(
            self.db.scalars(
                select(SampleEvent)
                .where(SampleEvent.sample_id == created_sample_id)
                .order_by(SampleEvent.event_year)
            )
        )

        self.assertEqual(first["created"], 1)
        self.assertEqual(second["created"], 0)
        self.assertEqual([event.event_year for event in events], [2018, 2020])
        self.assertEqual(
            [self.db.get(EventReview, event.event_id).review_status for event in events],
            ["DRAFT", "DRAFT"],
        )
        self.assertIsNone(self.db.get(ManualValidation, created_sample_id))
        with tempfile.TemporaryDirectory() as model_dir:
            with patch("backend.forest.model_labeling_service.FOREST_MODELS_DIR", Path(model_dir)):
                plan = build_model_labeling_plan(self.db, mining_run_id=run.run_id)
        self.assertEqual(plan["summary"]["events"], 2)
        self.assertEqual(plan["summary"]["incomplete_inputs"], 2)
        self.assertFalse(plan["summary"]["remoteclip_artifact_ready"])

    def test_model_labeling_is_idempotent_and_keeps_manual_review_draft(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            models_root = root / "models"
            exports_root = root / "exports"
            checkpoint = root / "remoteclip.pt"
            checkpoint.write_bytes(b"checkpoint")

            def write_artifact(directory: str, filename: str, manifest: dict) -> None:
                model_dir = models_root / directory / "v1"
                model_dir.mkdir(parents=True)
                artifact = model_dir / filename
                artifact.write_bytes(f"{directory}-artifact".encode("utf-8"))
                manifest.update(
                    {
                        "producer_version": "v1",
                        "artifact_sha256": file_sha256(artifact),
                        "class_order": ["Logging", "Wildfire"],
                    }
                )
                (model_dir / "manifest.json").write_text(
                    json.dumps(manifest),
                    encoding="utf-8",
                )

            write_artifact(
                "remoteclip_logreg",
                "classifier.joblib",
                {
                    "producer_name": "REMOTECLIP_LOGREG",
                    "feature_schema_version": "remoteclip-test-v1",
                    "feature_variant": "POST",
                    "embedding": {
                        "checkpoint_path": str(checkpoint),
                        "checkpoint_sha256": file_sha256(checkpoint),
                    },
                },
            )
            write_artifact(
                "xgboost",
                "model.json",
                {
                    "producer_name": "XGBOOST",
                    "feature_schema_version": "xgboost-test-v1",
                    "feature_names": ["ndvi_median_rel"],
                    "preprocessing": {"fill_values": {"ndvi_median_rel": 0.0}},
                },
            )

            item = sample("model-label-sample")
            item.source_format = "HANSEN_MINING"
            item.driver_primary = None
            event = SampleEvent(
                event_id=event_id_for(item.sample_id, "HANSEN", 2020),
                sample_id=item.sample_id,
                event_year=2020,
                source="HANSEN",
                loss_area_ha=5.0,
                loss_share=1.0,
                is_dominant=True,
            )
            run_id = "model-label-run"
            self.db.add_all(
                [
                    item,
                    event,
                    EventReview(event_id=event.event_id, review_status="DRAFT"),
                    SampleAssignment(
                        sample_id=item.sample_id,
                        assignment_type="hansen_mining_run",
                        assignment_name=run_id,
                    ),
                ]
            )
            for period in ("PRE", "POST"):
                geotiff = root / f"{period.lower()}.tif"
                preview = root / f"{period.lower()}.png"
                metadata = root / f"{period.lower()}.json"
                geotiff.write_bytes(b"geotiff")
                preview.write_bytes(b"preview")
                metadata.write_text("{}", encoding="utf-8")
                download_id = f"download-{period.lower()}"
                self.db.add_all(
                    [
                        SentinelDownload(
                            download_id=download_id,
                            sample_id=item.sample_id,
                            scene_id=f"scene-{period.lower()}",
                            period=period,
                            local_path=str(geotiff),
                            bbox_json="[]",
                            bands_json="[]",
                            metadata_json="{}",
                        ),
                        SampleEventDownload(
                            id=f"link-{period.lower()}",
                            event_id=event.event_id,
                            download_id=download_id,
                            period=period,
                        ),
                        DerivedPreview(
                            preview_id=f"preview-{period.lower()}",
                            download_id=download_id,
                            view="rgb",
                            png_path=str(preview),
                            metadata_path=str(metadata),
                        ),
                    ]
                )
            self.db.commit()

            class FakeRunner:
                def __init__(self, source_name: str, artifact: dict):
                    self.source_name = source_name
                    self.source_version = artifact["producer_version"]

                def predict(self, inference_item, should_cancel=None):
                    label = "Logging" if self.source_name == "REMOTECLIP_LOGREG" else "Wildfire"
                    probabilities = (
                        {"Logging": 0.8, "Wildfire": 0.2}
                        if label == "Logging"
                        else {"Logging": 0.25, "Wildfire": 0.75}
                    )
                    return Prediction(
                        label=label,
                        confidence=probabilities[label],
                        probabilities=probabilities,
                        input_fingerprint=stable_hash(
                            {
                                "event_id": inference_item.event.event_id,
                                "source": self.source_name,
                                "version": self.source_version,
                            }
                        ),
                        metadata={"test": True},
                    )

            factories = {
                source: (lambda artifact, source=source: FakeRunner(source, artifact))
                for source in ("REMOTECLIP_LOGREG", "XGBOOST")
            }
            with (
                patch("backend.forest.model_labeling_service.FOREST_MODELS_DIR", models_root),
                patch(
                    "backend.forest.model_labeling_service.importlib.util.find_spec",
                    return_value=object(),
                ),
                patch("backend.forest.model_inference_service.FOREST_EXPORTS_DIR", exports_root),
            ):
                first = execute_model_labeling(
                    self.db,
                    mining_run_id=run_id,
                    execution_id="first",
                    runner_factories=factories,
                )
                second = execute_model_labeling(
                    self.db,
                    mining_run_id=run_id,
                    execution_id="second",
                    runner_factories=factories,
                )
                refreshed_plan = build_model_labeling_plan(
                    self.db,
                    mining_run_id=run_id,
                )

            labels = list(self.db.scalars(select(EventLabel).order_by(EventLabel.source_name)))
            self.assertEqual(first["created"], 2)
            self.assertEqual(first["failed"], 0)
            self.assertEqual(second["created"], 0)
            self.assertEqual(second["reused"], 2)
            self.assertEqual([label.source_name for label in labels], ["REMOTECLIP_LOGREG", "XGBOOST"])
            self.assertEqual(
                [
                    (prediction["source_name"], prediction["label"])
                    for prediction in refreshed_plan["events"][0]["existing_predictions"]
                ],
                [("REMOTECLIP_LOGREG", "Logging"), ("XGBOOST", "Wildfire")],
            )
            self.assertIsNone(self.db.get(ManualValidation, item.sample_id))
            self.assertEqual(self.db.get(EventReview, event.event_id).review_status, "DRAFT")
            self.assertTrue((exports_root / "model_labeling" / "first" / "predictions.csv").exists())
            accepted = labels[0]
            accepted_payload = save_event_review(
                self.db,
                event.event_id,
                review_status="VALID",
                actor_name="tester",
                label=accepted.label,
                based_on_label_id=accepted.label_id,
            )
            self.assertEqual(accepted_payload["review"]["review_status"], "VALID")
            self.assertEqual(
                accepted_payload["review"]["labels"][-1]["based_on_label_id"],
                accepted.label_id,
            )

    def test_candidate_mining_uses_local_lossyear_tile(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            tile_path = temp_path / "60N_030E.tif"
            data = np.zeros((200, 200), dtype=np.uint8)
            data[70:130, 70:130] = 20
            with rasterio.open(
                tile_path,
                "w",
                driver="GTiff",
                width=data.shape[1],
                height=data.shape[0],
                count=1,
                dtype="uint8",
                crs="EPSG:4326",
                transform=from_bounds(34.0, 54.0, 36.0, 56.0, data.shape[1], data.shape[0]),
                nodata=0,
            ) as dataset:
                dataset.write(data, 1)
            self.db.add(
                HansenTile(
                    layer="lossyear",
                    tile_id="60N_030E",
                    version="GFC-test",
                    local_path=str(tile_path),
                )
            )
            self.db.commit()

            with patch("backend.forest.candidate_service.FOREST_EXPORTS_DIR", temp_path / "exports"):
                result = mine_hansen_candidates(
                    self.db,
                    CandidateMiningOptions(
                        target_count=1,
                        min_dominant_event_aoi_percent=0.5,
                        scan_resolution_m=300.0,
                    ),
                )

            self.assertEqual(result["selected_count"], 1)
            self.assertEqual(result["items"][0]["primary_event_year"], 2020)
            self.assertGreater(result["items"][0]["dominant_event_aoi_percent"], 0.5)

    def test_candidate_mining_never_converts_out_of_range_loss_code_to_future_year(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            tile_path = temp_path / "60N_030E.tif"
            data = np.full((200, 200), 85, dtype=np.uint8)
            with rasterio.open(
                tile_path,
                "w",
                driver="GTiff",
                width=data.shape[1],
                height=data.shape[0],
                count=1,
                dtype="uint8",
                crs="EPSG:4326",
                transform=from_bounds(34.0, 54.0, 36.0, 56.0, data.shape[1], data.shape[0]),
                nodata=0,
            ) as dataset:
                dataset.write(data, 1)
            self.db.add(
                HansenTile(
                    layer="lossyear",
                    tile_id="60N_030E",
                    version="GFC-test",
                    local_path=str(tile_path),
                )
            )
            self.db.commit()

            with patch("backend.forest.candidate_service.FOREST_EXPORTS_DIR", temp_path / "exports"):
                result = mine_hansen_candidates(
                    self.db,
                    CandidateMiningOptions(target_count=1, scan_resolution_m=300.0),
                )

            self.assertEqual(result["selected_count"], 0)
            self.assertEqual(result["items"], [])
            self.assertEqual(result["summary"]["rejections"]["TILE_SCAN_FAILED"], 1)

    def test_candidate_inventory_rejects_treecover_registered_as_lossyear(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tile_path = Path(temp_dir) / "treecover2000" / "60N_030E.tif"
            tile_path.parent.mkdir(parents=True)
            tile_path.touch()
            self.db.add(
                HansenTile(
                    layer="lossyear",
                    tile_id="60N_030E",
                    version="GFC-test",
                    local_path=str(tile_path),
                )
            )
            self.db.commit()

            inventory, checks = hansen_tile_inventory(self.db)

            self.assertEqual(inventory, {})
            self.assertEqual(checks[0]["status"], "invalid")
            self.assertIn("treecover2000", checks[0]["error"])

    def test_whole_hansen_tile_preview_is_cached_and_keeps_year_palette(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            tile_path = temp_path / "source" / "60N_030E.tif"
            tile_path.parent.mkdir(parents=True)
            data = np.zeros((40, 80), dtype=np.uint8)
            data[:, :40] = 20
            data[:, 40:] = 21
            with rasterio.open(
                tile_path,
                "w",
                driver="GTiff",
                width=data.shape[1],
                height=data.shape[0],
                count=1,
                dtype="uint8",
                crs="EPSG:4326",
                transform=from_bounds(30.0, 50.0, 40.0, 60.0, data.shape[1], data.shape[0]),
                nodata=0,
            ) as dataset:
                dataset.write(data, 1)

            item = sample("tile-preview-sample")
            item.lat = 55.0
            item.lon = 35.0
            self.db.add_all(
                [
                    item,
                    HansenTile(
                        layer="lossyear",
                        tile_id="60N_030E",
                        version="GFC-test",
                        local_path=str(tile_path),
                    ),
                ]
            )
            self.db.commit()

            preview_cache = temp_path / "cache" / "hansen"
            with (
                patch("backend.forest.hansen_service.FOREST_HANSEN_CACHE_DIR", preview_cache),
                patch(
                    "backend.forest.hansen_service.cache_url",
                    side_effect=lambda path: f"/cache/test/{Path(path).name}",
                ),
            ):
                first = hansen_tile_preview_payload(self.db, item, max_size=64)
                second = hansen_tile_preview_payload(self.db, item, max_size=64)

            png_path = preview_cache / "tile_previews" / "GFC-test" / "lossyear" / "60N_030E.png"
            with Image.open(png_path) as image:
                self.assertEqual(image.size, (64, 32))
            self.assertEqual(first["bbox"], [30.0, 50.0, 40.0, 60.0])
            self.assertEqual(first["preview_dimensions"], [64, 32])
            self.assertEqual([row["year"] for row in first["legend"]], [2020, 2021])
            self.assertFalse(first["cache_hit"])
            self.assertTrue(second["cache_hit"])
            self.assertTrue(first["png_url"].startswith("/cache/test/60N_030E.png?v="))

    def test_candidate_hansen_preview_is_clipped_to_aoi_and_cached(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            tile_path = temp_path / "source" / "60N_030E.tif"
            tile_path.parent.mkdir(parents=True)
            data = np.full((400, 400), 20, dtype=np.uint8)
            data[:, 200:] = 85
            with rasterio.open(
                tile_path,
                "w",
                driver="GTiff",
                width=data.shape[1],
                height=data.shape[0],
                count=1,
                dtype="uint8",
                crs="EPSG:4326",
                transform=from_bounds(34.0, 54.0, 36.0, 56.0, data.shape[1], data.shape[0]),
                nodata=0,
            ) as dataset:
                dataset.write(data, 1)

            candidate = HansenCandidate(
                candidate_id="candidate-preview",
                algorithm_version=ALGORITHM_VERSION,
                hansen_dataset_version="GFC-test",
                lat=55.0,
                lon=35.0,
                aoi_size_m=2000.0,
                aoi_bbox_json="[34.98, 54.98, 35.02, 55.02]",
                primary_event_year=2020,
                event_years_json="[2020]",
                has_multiple_events=False,
                component_area_ha=20.0,
                dominant_loss_area_ha=20.0,
                dominant_loss_fraction=1.0,
                total_loss_area_ha=20.0,
                post_2016_loss_aoi_percent=5.0,
                dominant_event_aoi_percent=5.0,
                pixel_count=200,
                source_tile_ids_json='["60N_030E"]',
                component_key="60N_030E:2020:preview",
                metadata_json=json.dumps({"min_event_year": 2016}),
            )
            self.db.add_all(
                [
                    candidate,
                    HansenTile(
                        layer="lossyear",
                        tile_id="60N_030E",
                        version="GFC-test",
                        local_path=str(tile_path),
                    ),
                ]
            )
            self.db.commit()

            preview_cache = temp_path / "cache" / "hansen"
            with (
                patch(
                    "backend.forest.candidate_service.FOREST_HANSEN_CACHE_DIR",
                    preview_cache,
                ),
                patch(
                    "backend.forest.candidate_service.cache_url",
                    side_effect=lambda path: f"/cache/test/{Path(path).name}",
                ),
            ):
                first = candidate_hansen_preview_payload(self.db, candidate.candidate_id)
                second = candidate_hansen_preview_payload(self.db, candidate.candidate_id)

            png_path = (
                preview_cache
                / "candidate_previews"
                / "GFC-test"
                / candidate.candidate_id
                / "all_loss.png"
            )
            with Image.open(png_path) as image:
                self.assertGreater(image.width, 0)
                self.assertGreater(image.height, 0)
                self.assertGreater(image.getchannel("A").getextrema()[1], 0)
            self.assertEqual([row["year"] for row in first["legend"]], [2020])
            self.assertFalse(first["cache_hit"])
            self.assertTrue(second["cache_hit"])
            self.assertEqual(first["candidate_id"], candidate.candidate_id)
            self.assertTrue(first["png_url"].startswith("/cache/test/all_loss.png?v="))

    def test_candidate_mask_job_is_idempotent_and_preserves_event_links(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            tile_path = temp_path / "source" / "60N_030E.tif"
            tile_path.parent.mkdir(parents=True)
            data = np.zeros((400, 400), dtype=np.uint8)
            data[170:230, 170:230] = 20
            with rasterio.open(
                tile_path,
                "w",
                driver="GTiff",
                width=data.shape[1],
                height=data.shape[0],
                count=1,
                dtype="uint8",
                crs="EPSG:4326",
                transform=from_bounds(34.0, 54.0, 36.0, 56.0, data.shape[1], data.shape[0]),
                nodata=0,
            ) as dataset:
                dataset.write(data, 1)

            run = HansenCandidateRun(
                run_id="candidate-mask-run",
                algorithm_name=ALGORITHM_NAME,
                algorithm_version=ALGORITHM_VERSION,
                hansen_dataset_version="GFC-test",
                status="APPLIED",
                target_count=1,
                selected_count=1,
                created_sample_count=1,
                config_json="{}",
            )
            item = sample("candidate-mask-sample")
            item.source_format = "HANSEN_MINING"
            item.driver_primary = None
            event = SampleEvent(
                event_id=event_id_for(item.sample_id, "HANSEN", 2020),
                sample_id=item.sample_id,
                event_year=2020,
                source="HANSEN",
                loss_area_ha=20.0,
                loss_share=1.0,
                is_dominant=True,
            )
            download = SentinelDownload(
                download_id="candidate-mask-download",
                sample_id=item.sample_id,
                scene_id="candidate-mask-scene",
                period="POST",
                local_path=str(temp_path / "sentinel.tif"),
                bbox_json="[]",
                bands_json="[]",
                metadata_json="{}",
            )
            candidate = HansenCandidate(
                candidate_id="candidate-mask",
                algorithm_version=ALGORITHM_VERSION,
                hansen_dataset_version="GFC-test",
                lat=55.0,
                lon=35.0,
                aoi_size_m=2000.0,
                aoi_bbox_json="[34.98, 54.98, 35.02, 55.02]",
                primary_event_year=2020,
                event_years_json="[2020]",
                has_multiple_events=False,
                component_area_ha=20.0,
                dominant_loss_area_ha=20.0,
                dominant_loss_fraction=1.0,
                total_loss_area_ha=20.0,
                post_2016_loss_aoi_percent=5.0,
                dominant_event_aoi_percent=5.0,
                pixel_count=200,
                source_tile_ids_json='["60N_030E"]',
                component_key="60N_030E:2020:mask",
                created_sample_id=item.sample_id,
                metadata_json=json.dumps(
                    {
                        "min_event_year": 2016,
                        "histogram": [
                            {"year": 2020, "pixels": 200, "loss_area_ha": 20.0},
                        ],
                    }
                ),
            )
            self.db.add_all(
                [
                    run,
                    item,
                    event,
                    EventReview(event_id=event.event_id, review_status="DRAFT"),
                    download,
                    SampleEventDownload(
                        id="candidate-mask-link",
                        event_id=event.event_id,
                        download_id=download.download_id,
                        period="POST",
                    ),
                    candidate,
                    HansenCandidateRunItem(
                        id="candidate-mask-run-item",
                        run_id=run.run_id,
                        candidate_id=candidate.candidate_id,
                        decision="SELECTED",
                        rank=1,
                    ),
                    HansenTile(
                        layer="lossyear",
                        tile_id="60N_030E",
                        version="GFC-test",
                        local_path=str(tile_path),
                    ),
                ]
            )
            self.db.commit()

            cache_root = temp_path / "cache" / "hansen"
            exports_root = temp_path / "exports"
            with (
                patch(
                    "backend.forest.candidate_service.FOREST_HANSEN_CACHE_DIR",
                    cache_root,
                ),
                patch(
                    "backend.forest.candidate_service.FOREST_EXPORTS_DIR",
                    exports_root,
                ),
            ):
                first = generate_candidate_masks_for_run(
                    self.db,
                    run.run_id,
                    execution_id="first",
                )
                second = generate_candidate_masks_for_run(
                    self.db,
                    run.run_id,
                    execution_id="second",
                )
                status = candidate_mask_run_status(self.db, run.run_id)

            analyses = list(
                self.db.scalars(
                    select(HansenAnalysis).where(HansenAnalysis.sample_id == item.sample_id)
                )
            )
            masks = list(
                self.db.scalars(
                    select(HansenMask).where(HansenMask.sample_id == item.sample_id)
                )
            )
            self.assertEqual(first["created"], 1)
            self.assertEqual(first["events_modified"], 0)
            self.assertEqual(second["created"], 0)
            self.assertEqual(second["reused"], 1)
            self.assertEqual(status, {"materialized": 1, "ready": 1, "pending": 0})
            self.assertEqual(len(analyses), 1)
            self.assertEqual(analyses[0].analysis_area_type, "candidate_aoi")
            self.assertEqual(
                {mask.mask_type for mask in masks},
                {"aoi_all_loss", "aoi_dominant_year"},
            )
            self.assertEqual(self.db.get(SampleEvent, event.event_id).event_year, 2020)
            self.assertIsNone(self.db.get(SampleEvent, event.event_id).source_analysis_id)
            self.assertEqual(self.db.get(EventReview, event.event_id).review_status, "DRAFT")
            self.assertIsNotNone(self.db.get(SampleEventDownload, "candidate-mask-link"))
            for mask in masks:
                path = Path(mask.local_path)
                if not path.is_absolute():
                    path = TOOL_ROOT.parents[1] / path
                self.assertTrue(path.exists())
                self.assertTrue(path.with_suffix(".png").exists())

    def test_db_sync_can_insert_versioned_label_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            incoming_path = Path(temp_dir) / "incoming.sqlite"
            incoming_engine = create_engine(f"sqlite+pysqlite:///{incoming_path.as_posix()}", future=True)
            Base.metadata.create_all(incoming_engine)
            with Session(incoming_engine) as incoming_db:
                incoming_db.add(
                    LabelRun(
                        label_run_id="model-run-1",
                        producer_type="MODEL",
                        producer_name="XGBOOST",
                        producer_version="xgboost_v1",
                        model_artifact_path="Scripts/ResearchTool/Data/forest/models/xgboost/v1/model.json",
                        model_artifact_sha256="a" * 64,
                        feature_schema_version="hansen_v1",
                        config_json="{}",
                        status="READY",
                    )
                )
                incoming_db.commit()

            summary: dict[str, Counter] = defaultdict(Counter)
            operations: list[dict] = []
            connection = sqlite3.connect(incoming_path)
            try:
                connection.row_factory = sqlite3.Row
                compare_portable_tables(
                    self.db,
                    connection,
                    summary=summary,
                    operations=operations,
                )
            finally:
                connection.close()
            safe = [row for row in operations if row["safe"]]
            self.assertEqual(len(safe), 1)
            self.assertEqual(safe[0]["entity"], "label_runs")
            apply_operation(self.db, safe[0])
            self.db.commit()
            self.assertEqual(self.db.get(LabelRun, "model-run-1").producer_name, "XGBOOST")
            incoming_engine.dispose()


if __name__ == "__main__":
    unittest.main()

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
from backend.forest.candidate_service import (  # noqa: E402
    ALGORITHM_NAME,
    ALGORITHM_VERSION,
    CandidateMiningOptions,
    materialize_candidates,
    mine_hansen_candidates,
    stable_id,
)
from backend.forest.models import (  # noqa: E402
    Base,
    EventReview,
    HansenCandidate,
    HansenCandidateRun,
    HansenCandidateRunItem,
    HansenAnalysis,
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
    plan_sentinel_batch,
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
                max_qc_scenes=5,
            ),
        )
        self.assertEqual(plan["summary"]["samples_actionable"], 1)
        self.assertEqual(plan["summary"]["operation_rows"], 9)

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

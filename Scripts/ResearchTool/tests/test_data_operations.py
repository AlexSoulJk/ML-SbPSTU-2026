from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session


TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from backend.forest.event_service import (  # noqa: E402
    backfill_hansen_events,
    event_id_for,
    upsert_event_download_link,
)
from backend.forest.models import (  # noqa: E402
    Base,
    HansenAnalysis,
    Sample,
    SampleEvent,
    SampleEventDownload,
    SentinelDownload,
)
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
        self.db = Session(self.engine)

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


if __name__ == "__main__":
    unittest.main()

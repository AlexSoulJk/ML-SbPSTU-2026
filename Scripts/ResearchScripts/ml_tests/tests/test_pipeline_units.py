from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ml_tests.dataset import season_for_datetime, validation_cohort
from ml_tests.embeddings import EmbeddingCache, l2_normalize
from ml_tests.event_links import apply_dominant_event_backfill, plan_dominant_event_backfill
from ml_tests.experiments import records_for_variant
from ml_tests.reporting import baseline_metrics_html
from ml_tests.splits import assert_no_group_leakage, build_saved_splits

from ResearchTool.backend.forest.models import (
    Base,
    Sample,
    SampleEvent,
    SampleEventDownload,
    SentinelDownload,
)


def sample(sample_id: str) -> Sample:
    return Sample(
        sample_id=sample_id,
        source_file="test.geojson",
        source_format="geojson",
        lat=50.0,
        lon=40.0,
        driver_primary="Logging",
        raw_properties_json="{}",
    )


class EventBackfillTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite:///:memory:", future=True)
        Base.metadata.create_all(self.engine)

    def test_dominant_backfill_is_idempotent(self) -> None:
        with Session(self.engine) as db:
            db.add(sample("sample-1"))
            db.add_all(
                [
                    SampleEvent(
                        event_id="event-2020",
                        sample_id="sample-1",
                        event_year=2020,
                        source="HANSEN",
                        is_dominant=False,
                    ),
                    SampleEvent(
                        event_id="event-2021",
                        sample_id="sample-1",
                        event_year=2021,
                        source="HANSEN",
                        is_dominant=True,
                    ),
                    SentinelDownload(
                        download_id="download-1",
                        sample_id="sample-1",
                        scene_id="scene-1",
                        period="POST",
                        local_path="image.tif",
                        bbox_json="[]",
                        bands_json="[]",
                        metadata_json="{}",
                    ),
                ]
            )
            db.commit()

            plan = plan_dominant_event_backfill(db)
            self.assertEqual(plan["counts"]["would_link"], 1)
            result = apply_dominant_event_backfill(db, plan["rows"])
            db.commit()
            self.assertEqual(result["counts"]["linked"], 1)

            second_plan = plan_dominant_event_backfill(db)
            self.assertEqual(second_plan["counts"]["already_linked"], 1)
            self.assertNotIn("would_link", second_plan["counts"])

    def test_existing_link_is_preserved(self) -> None:
        with Session(self.engine) as db:
            db.add(sample("sample-2"))
            db.add(
                SampleEvent(
                    event_id="event-existing",
                    sample_id="sample-2",
                    event_year=2022,
                    source="HANSEN",
                    is_dominant=True,
                )
            )
            db.add(
                SentinelDownload(
                    download_id="download-existing",
                    sample_id="sample-2",
                    scene_id="scene-existing",
                    period="PRE",
                    local_path="image.tif",
                    bbox_json="[]",
                    bands_json="[]",
                    metadata_json="{}",
                )
            )
            db.add(
                SampleEventDownload(
                    id="link-existing",
                    event_id="event-existing",
                    download_id="download-existing",
                    period="PRE",
                )
            )
            db.commit()
            plan = plan_dominant_event_backfill(db)
            self.assertEqual(plan["counts"], {"already_linked": 1})


class DatasetHelpersTests(unittest.TestCase):
    def test_validation_cohort_requires_missing_row_for_weak(self) -> None:
        self.assertEqual(validation_cohort(None), "NO_MANUAL_WEAK")
        self.assertEqual(validation_cohort("Valid"), "MANUAL_VALID")
        self.assertEqual(validation_cohort("Too old"), "MANUAL_TOO_OLD")

    def test_season_assignment_uses_configured_windows(self) -> None:
        from datetime import datetime

        windows = {
            "spring": ["04-15", "05-31"],
            "early_summer": ["06-01", "07-15"],
        }
        self.assertEqual(season_for_datetime(datetime(2024, 5, 1), windows), "spring")
        self.assertEqual(
            season_for_datetime(datetime(2024, 7, 20), windows),
            "outside_defined_seasons",
        )


class EmbeddingCacheTests(unittest.TestCase):
    def test_cache_round_trip_and_corrupt_entry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = EmbeddingCache(directory, embedder_fingerprint="model-v1", policy="read_write")
            row = {"download_id": "d1", "view": "rgb"}
            key = cache.key_for(row, "source-v1")
            expected = l2_normalize(np.array([1.0, 2.0, 3.0], dtype=np.float32))
            cache.write(key, expected, {"source_fingerprint": "source-v1"})
            np.testing.assert_allclose(cache.read(key), expected)

            cache.path_for(key).write_bytes(b"incomplete")
            self.assertIsNone(cache.read(key))


class SplitAndWeightTests(unittest.TestCase):
    def test_group_splits_do_not_leak(self) -> None:
        records = []
        for index in range(12):
            records.append(
                {
                    "record_id": f"event-{index}",
                    "sample_id": f"sample-{index}",
                    "label": "Logging" if index < 6 else "Wildfire",
                }
            )
        splits = build_saved_splits(
            records,
            requested_n_splits=3,
            seeds=[42],
            include_ordinary=True,
        )
        assert_no_group_leakage(splits)
        self.assertEqual(splits["group_n_splits"], 3)

    def test_multi_event_weights_sum_to_one(self) -> None:
        base = {
            "label": "Logging",
            "validation_cohort": "MANUAL_VALID",
            "vectors": {"POST": np.array([1.0, 0.0], dtype=np.float32)},
        }
        records = records_for_variant(
            [
                {**base, "record_id": "e1", "event_id": "e1", "sample_id": "s1"},
                {**base, "record_id": "e2", "event_id": "e2", "sample_id": "s1"},
                {**base, "record_id": "e3", "event_id": "e3", "sample_id": "s2"},
            ],
            "POST",
        )
        weights = {}
        for record in records:
            weights[record["sample_id"]] = weights.get(record["sample_id"], 0.0) + record["sample_weight"]
        self.assertEqual(weights, {"s1": 1.0, "s2": 1.0})


class NotebookTests(unittest.TestCase):
    def test_notebook_is_thin_and_valid_json(self) -> None:
        path = Path(__file__).resolve().parents[1] / "forest_ml_experiments.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        self.assertNotIn("def ", code)
        self.assertIn("ForestMLPipeline", code)
        self.assertIn("REUSE_SAVED_BASELINE", code)
        self.assertIn("baseline_metrics_html", code)
        self.assertEqual(notebook["nbformat"], 4)

    def test_metrics_table_contains_actual_values(self) -> None:
        rendered = baseline_metrics_html(
            [
                {
                    "feature_variant": "POST",
                    "mode": "MANUAL_ONLY",
                    "split_strategy": "stratified_group",
                    "fold_runs": 5,
                    "accuracy_mean": 0.75,
                    "accuracy_std": 0.05,
                    "balanced_accuracy_mean": 0.7,
                    "balanced_accuracy_std": 0.04,
                    "f1_mean": 0.8,
                    "f1_std": 0.03,
                    "roc_auc_mean": 0.82,
                    "roc_auc_std": 0.02,
                }
            ]
        )
        self.assertIn("0.750 +/- 0.050", rendered)
        self.assertIn("POST", rendered)


if __name__ == "__main__":
    unittest.main()


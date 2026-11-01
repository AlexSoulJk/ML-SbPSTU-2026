from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ml_tests.dataset import season_for_datetime, validation_cohort
from ml_tests.artifacts import export_remoteclip_classifier
from ml_tests.config import ExperimentConfig
from ml_tests.embeddings import EmbeddingCache, l2_normalize
from ml_tests.event_links import apply_dominant_event_backfill, plan_dominant_event_backfill
from ml_tests.experiments import records_for_variant
from ml_tests.latent_diagnostics import (
    _align_labels,
    _class_conditional_scores,
    _knn_scores,
)
from ml_tests.reporting import baseline_metrics_html
from ml_tests.sample_evaluation import run_valid_post_clustering
from ml_tests.splits import assert_no_group_leakage, build_saved_splits

from ResearchTool.backend.forest.models import (
    Base,
    Sample,
    SampleEvent,
    SampleEventDownload,
    SentinelDownload,
)
from ResearchTool.backend.forest.model_labeling_service import model_artifact_status


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


class ValidPostSampleEvaluationTests(unittest.TestCase):
    def test_clustering_uses_only_manual_valid_post_events(self) -> None:
        generator = np.random.default_rng(42)
        manifest: list[dict] = []
        embeddings: dict[str, np.ndarray] = {}
        for index in range(30):
            download_id = f"download-{index}"
            center = np.array([1, 1, 1, 0, 0, 0], dtype=np.float32)
            if index >= 15:
                center = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32)
            embeddings[f"{download_id}:rgb"] = center + generator.normal(0, 0.08, 6)
            manifest.append(
                {
                    "eligible": True,
                    "sample_id": f"sample-{index}",
                    "event_id": f"event-{index}",
                    "event_year": 2020 + index % 3,
                    "event_loss_area_ha": 10.0,
                    "event_loss_share": 0.5,
                    "event_is_dominant": True,
                    "is_multi_event_area": False,
                    "label": "Logging" if index < 15 else "Wildfire",
                    "validation_cohort": "MANUAL_VALID",
                    "period": "POST",
                    "download_id": download_id,
                    "view": "rgb",
                    "season": "late_summer",
                }
            )
        manifest.append(
            {
                **manifest[0],
                "sample_id": "not-valid-sample",
                "event_id": "not-valid-event",
                "download_id": "not-valid-download",
                "validation_cohort": "NO_MANUAL_WEAK",
            }
        )
        embeddings["not-valid-download:rgb"] = np.ones(6, dtype=np.float32)

        result = run_valid_post_clustering(
            manifest,
            embeddings,
            k_min=2,
            k_max=5,
            stability_runs=4,
            stability_sample_fraction=0.8,
            hdbscan_parameter_grid=[(5, 3), (8, 5)],
        )

        self.assertEqual(result["summary"]["events"], 30)
        self.assertEqual(result["summary"]["validation_cohort"], "MANUAL_VALID")
        self.assertEqual(result["summary"]["period"], "POST")
        self.assertEqual(result["summary"]["silhouette_k"], 2)
        self.assertEqual(len(result["event_rows"]), 30)
        np.testing.assert_allclose(
            np.linalg.norm(result["dataset"]["matrix"], axis=1),
            np.ones(30),
            atol=1e-6,
        )

    def test_loo_knownness_excludes_self_and_class_similarity_is_labeled(self) -> None:
        matrix = np.asarray(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
            ],
            dtype=np.float32,
        )
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        scores = _knn_scores(matrix, matrix, k_values=(1, 2, 3), leave_one_out=True)
        self.assertLess(scores[0]["nearest_similarity"], 1.0)
        self.assertAlmostEqual(
            scores[0]["knownness_k1"],
            float(matrix[0] @ matrix[1]),
            places=6,
        )
        conditional = _class_conditional_scores(
            matrix,
            matrix,
            ["Logging", "Logging", "Wildfire", "Wildfire"],
            leave_one_out=True,
        )
        self.assertEqual(conditional[0]["nearest_class_prediction"], "Logging")
        self.assertEqual(conditional[2]["nearest_class_prediction"], "Wildfire")

    def test_cluster_alignment_removes_label_permutation(self) -> None:
        reference = np.asarray([0, 0, 1, 1, 2, 2])
        predicted = np.asarray([2, 2, 0, 0, 1, 1])
        np.testing.assert_array_equal(_align_labels(reference, predicted), reference)


class ModelArtifactTests(unittest.TestCase):
    def test_remoteclip_artifact_round_trip_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "remoteclip.pt"
            checkpoint.write_bytes(b"test-checkpoint")
            config_path = root / "config.json"
            config_path.write_text("{}", encoding="utf-8")
            config = ExperimentConfig(
                path=config_path,
                profile_name="quick",
                data={
                    "paths": {"checkpoint": str(checkpoint)},
                    "embedding": {
                        "model_name": "ViT-B-32",
                        "preprocessing_version": "test-preprocess-v1",
                        "normalization_version": "l2_v1",
                    },
                    "classifier": {
                        "class_weight": "balanced",
                        "max_iter": 200,
                        "solver": "liblinear",
                        "random_state": 42,
                    },
                    "profiles": {"quick": {"n_splits": 2, "seeds": [42]}},
                },
            )
            event_records = []
            for index, (label, vector) in enumerate(
                [
                    ("Logging", [1.0, 0.0]),
                    ("Logging", [0.9, 0.1]),
                    ("Wildfire", [0.0, 1.0]),
                    ("Wildfire", [0.1, 0.9]),
                ]
            ):
                event_records.append(
                    {
                        "record_id": f"event-{index}",
                        "event_id": f"event-{index}",
                        "sample_id": f"sample-{index}",
                        "label": label,
                        "validation_cohort": "MANUAL_VALID",
                        "vectors": {"POST": np.asarray(vector, dtype=np.float32)},
                    }
                )
            baseline_metrics = [
                {
                    "feature_variant": "POST",
                    "mode": "MANUAL_ONLY",
                    "split_strategy": "stratified_group",
                    "balanced_accuracy_mean": 0.8,
                    "roc_auc_mean": 0.9,
                    "f1_mean": 0.75,
                }
            ]
            models_root = root / "models"
            with (
                patch("ml_tests.artifacts.FOREST_MODELS_DIR", models_root),
                patch(
                    "ResearchTool.backend.forest.model_labeling_service.FOREST_MODELS_DIR",
                    models_root,
                ),
            ):
                result = export_remoteclip_classifier(
                    event_records,
                    baseline_metrics,
                    config,
                    {"sha256": "database-test-sha"},
                    version="test-v1",
                )
                artifact_status = model_artifact_status("REMOTECLIP_LOGREG", "test-v1")

            model_dir = models_root / "remoteclip_logreg" / "test-v1"
            artifact = joblib.load(model_dir / "classifier.joblib")
            manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(artifact["feature_variant"], "POST")
            self.assertEqual(artifact["class_order"], ["Logging", "Wildfire"])
            self.assertEqual(manifest["artifact_sha256"], result["artifact_sha256"])
            self.assertEqual(manifest["required_periods"], ["POST"])
            self.assertEqual(manifest["round_trip_smoke_test"]["status"], "passed")
            self.assertEqual(result["training"]["records"], 4)
            self.assertTrue(artifact_status["ready"], artifact_status["errors"])


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
        self.assertIn("SAVE_REMOTECLIP_ARTIFACT", code)
        self.assertIn("plan_remoteclip_model_export", code)
        self.assertEqual(notebook["nbformat"], 4)

    def test_sample_evaluation_notebook_contains_latent_diagnostics(self) -> None:
        path = Path(__file__).resolve().parents[1] / "valid_post_sample_evaluation.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        markdown = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
        )
        self.assertNotIn("def ", code)
        self.assertIn("run_latent_diagnostics", code)
        self.assertIn("latent_event_level_table.csv", markdown)
        self.assertIn("Manual-vs-Hansen domain shift", markdown)
        self.assertIn("Combined UNKNOWN and AMBIGUOUS reject", markdown)
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


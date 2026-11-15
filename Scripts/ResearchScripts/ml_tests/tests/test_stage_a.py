from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ml_tests.config import stable_json_hash
from ml_tests.splits import (
    assert_no_group_leakage,
    build_constrained_grouped_splits,
    build_saved_splits,
)
from ml_tests.stage_a_dataset import build_current_event_manifest, load_reviewed_baseline_cohort
from ml_tests.stage_a_models import run_cohort_size_experiment
from ResearchTool.backend.forest.models import (
    Base,
    EventLabel,
    EventReview,
    ManualValidation,
    Sample,
    SampleEvent,
)


def _remoteclip_records() -> list[dict]:
    rows = []
    for index in range(24):
        label = "Logging" if index < 12 else "Wildfire"
        rows.append(
            {
                "record_id": f"event-{index:02d}",
                "event_id": f"event-{index:02d}",
                "sample_id": f"sample-{index:02d}",
                "event_year": 2016 + index % 8,
                "label": label,
                "features": np.asarray(
                    [float(index), float(index % 3), 0.0 if label == "Logging" else 1.0],
                    dtype=np.float32,
                ),
                "sample_weight": 1.0,
            }
        )
    return rows


def _xgboost_records() -> list[dict]:
    return [
        {
            **{key: value for key, value in row.items() if key != "features"},
            "features": {
                "f1": float(row["features"][0]),
                "f2": float(row["features"][2]),
            },
        }
        for row in _remoteclip_records()
    ]


class StageAUnitTests(unittest.TestCase):
    def test_manifest_unites_legacy_and_event_level_human_truth(self) -> None:
        engine = create_engine("sqlite:///:memory:", future=True)
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine, future=True)

        def sample(sample_id: str, driver: str | None) -> Sample:
            return Sample(
                sample_id=sample_id,
                source_file="legacy.csv" if driver else "hansen_candidate_v1",
                source_format="csv" if driver else "HANSEN_MINING",
                lat=50.0,
                lon=40.0,
                driver_primary=driver,
                raw_properties_json="{}",
            )

        def event(event_id: str, sample_id: str, year: int) -> SampleEvent:
            return SampleEvent(
                event_id=event_id,
                sample_id=sample_id,
                event_year=year,
                source="HANSEN",
                loss_area_ha=1.0,
                loss_share=1.0,
                is_dominant=True,
            )

        with Session() as db:
            db.add_all(
                [
                    sample("legacy", "Logging"),
                    sample("new", None),
                    sample("override", "Logging"),
                    sample("rejected", "Logging"),
                ]
            )
            db.add_all(
                [
                    event("legacy-event", "legacy", 2019),
                    event("new-event", "new", 2020),
                    event("override-event", "override", 2021),
                    event("rejected-event", "rejected", 2022),
                ]
            )
            db.add_all(
                [
                    ManualValidation(sample_id="legacy", validation="Valid"),
                    ManualValidation(sample_id="override", validation="Valid"),
                    ManualValidation(sample_id="rejected", validation="Valid"),
                ]
            )
            db.add_all(
                [
                    EventLabel(
                        label_id="new-label",
                        sample_id="new",
                        event_id="new-event",
                        source_type="HUMAN",
                        source_name="test",
                        source_version="1",
                        label="Wildfire",
                        input_fingerprint="new",
                    ),
                    EventLabel(
                        label_id="override-label",
                        sample_id="override",
                        event_id="override-event",
                        source_type="HUMAN",
                        source_name="test",
                        source_version="1",
                        label="Wildfire",
                        input_fingerprint="override",
                    ),
                ]
            )
            db.add_all(
                [
                    EventReview(
                        event_id="new-event",
                        review_status="VALID",
                        human_label_id="new-label",
                    ),
                    EventReview(
                        event_id="override-event",
                        review_status="VALID",
                        human_label_id="override-label",
                    ),
                    EventReview(event_id="rejected-event", review_status="REJECTED"),
                ]
            )
            db.commit()
            rows, images, audit = build_current_event_manifest(
                db,
                database_fingerprint={"sha256": "test"},
                virtual_links=[
                    {
                        "event_id": "legacy-event",
                        "download_id": "legacy-download",
                        "legacy_period": "PRE",
                    }
                ],
            )

        by_id = {row["event_id"]: row for row in rows}
        self.assertEqual(set(by_id), {"legacy-event", "new-event", "override-event"})
        self.assertEqual(by_id["legacy-event"]["human_label"], "Logging")
        self.assertEqual(
            by_id["legacy-event"]["ground_truth_source"],
            "SAMPLE_MANUAL_VALIDATION",
        )
        self.assertEqual(by_id["override-event"]["human_label"], "Wildfire")
        self.assertEqual(by_id["override-event"]["ground_truth_source"], "EVENT_REVIEW")
        self.assertEqual(images, [])
        self.assertEqual(
            audit["ground_truth_source_counts"],
            {"EVENT_REVIEW": 2, "SAMPLE_MANUAL_VALIDATION": 1},
        )
        self.assertEqual(audit["virtual_dominant_backfill_links"], 1)

    def test_stage_a_notebook_is_thin_and_safe_to_reopen(self) -> None:
        notebook_path = Path(__file__).resolve().parents[1] / "pipeline_v2_stage_a.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
        )
        self.assertIn("RUN_STAGE_A = False", source)
        self.assertLess(source.index("sys.path.insert"), source.index("from ml_tests.stage_a"))
        self.assertIn("not_ready_df", source)
        self.assertIn("profile_split_df", source)
        self.assertIn("Aggregate confusion matrices", source)
        self.assertNotIn("def ", source)

    def test_constrained_splits_keep_both_classes_in_old_train(self) -> None:
        records = _remoteclip_records()
        old_ids = {
            row["event_id"]
            for row in records
            if int(row["event_id"].split("-")[-1]) in set(range(6)) | set(range(12, 18))
        }
        splits = build_constrained_grouped_splits(
            records,
            old_record_ids=old_ids,
            required_old_record_id_sets={"remoteclip": old_ids, "xgboost": old_ids},
            requested_n_splits=4,
            seeds=[0, 1],
        )
        repeated = build_constrained_grouped_splits(
            records,
            old_record_ids=old_ids,
            required_old_record_id_sets={"remoteclip": old_ids, "xgboost": old_ids},
            requested_n_splits=4,
            seeds=[0, 1],
        )
        self.assertEqual(splits, repeated)
        by_id = {row["event_id"]: row for row in records}
        self.assertEqual(len(splits["rows"]), 8)
        for split in splits["rows"]:
            old_train = [
                by_id[event_id]
                for event_id in split["train_record_ids"]
                if event_id in old_ids
            ]
            self.assertEqual({row["label"] for row in old_train}, {"Logging", "Wildfire"})

    def test_grouped_splits_allow_mixed_labels_inside_one_sample(self) -> None:
        records = _remoteclip_records()
        records[12]["sample_id"] = records[0]["sample_id"]
        splits = build_saved_splits(
            records,
            requested_n_splits=4,
            seeds=[11],
            include_ordinary=False,
        )
        assert_no_group_leakage(splits)
        for row in splits["rows"]:
            self.assertFalse(set(row["train_groups"]) & set(row["test_groups"]))

    def test_grouped_splits_are_reused_by_both_cohort_modes(self) -> None:
        records = _remoteclip_records()
        splits = build_saved_splits(
            records,
            requested_n_splits=4,
            seeds=[3],
            include_ordinary=False,
        )
        assert_no_group_leakage(splits)
        old_ids = {
            row["event_id"]
            for row in records
            if int(row["event_id"].split("-")[-1]) in set(range(8)) | set(range(12, 20))
        }
        result = run_cohort_size_experiment(
            records,
            old_event_ids=old_ids,
            saved_splits=splits,
            model_name="REMOTECLIP_LOGREG",
            remoteclip_classifier_config={
                "class_weight": "balanced",
                "max_iter": 1000,
                "solver": "liblinear",
                "random_state": 3,
            },
        )
        self.assertEqual(len(result["fold_metrics"]), 8)
        by_fold = {}
        for row in result["fold_metrics"]:
            by_fold.setdefault((row["seed"], row["fold"]), []).append(row)
        self.assertTrue(
            all(len({item["test_events"] for item in rows}) == 1 for rows in by_fold.values())
        )
        self.assertEqual(
            {row["cohort_mode"] for row in result["summary_metrics"]},
            {"OLD_COHORT_CV", "FULL_COHORT_CV"},
        )

    def test_xgboost_cohort_experiment_uses_same_test_folds(self) -> None:
        template = _remoteclip_records()
        splits = build_saved_splits(
            template,
            requested_n_splits=3,
            seeds=[7],
            include_ordinary=False,
        )
        old_ids = {
            row["event_id"]
            for row in template
            if int(row["event_id"].split("-")[-1]) in set(range(9)) | set(range(12, 21))
        }
        result = run_cohort_size_experiment(
            _xgboost_records(),
            old_event_ids=old_ids,
            saved_splits=splits,
            model_name="XGBOOST",
            xgboost_feature_names=["f1", "f2"],
            xgboost_parameters={
                "n_estimators": 5,
                "max_depth": 2,
                "learning_rate": 0.1,
                "tree_method": "hist",
                "random_state": 7,
                "n_jobs": 1,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
            },
        )
        self.assertEqual(len(result["fold_metrics"]), 6)
        self.assertTrue(all(row["status"] == "ok" for row in result["fold_metrics"]))

    def test_reviewed_baseline_is_reconciled_with_current_labels(self) -> None:
        current = [
            {"event_id": "a", "sample_id": "s1", "event_year": 2020, "human_label": "Logging"},
            {"event_id": "b", "sample_id": "s2", "event_year": 2021, "human_label": "Wildfire"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "event_level.csv"
            with path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("event_id", "sample_id", "event_year", "review_status", "human_label"),
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {"event_id": "a", "sample_id": "s1", "event_year": 2020, "review_status": "VALID", "human_label": "Logging"},
                        {"event_id": "b", "sample_id": "s2", "event_year": 2021, "review_status": "VALID", "human_label": "Logging"},
                        {"event_id": "c", "sample_id": "s3", "event_year": 2022, "review_status": "REJECTED", "human_label": "Wildfire"},
                    ]
                )
            result = load_reviewed_baseline_cohort(path, current)
        self.assertEqual(result["matched_event_ids"], ["a"])
        self.assertEqual(result["status_counts"], {"LABEL_CHANGED": 1, "MATCHED": 1})

    def test_stage_a_config_is_final_profile_5_by_10(self) -> None:
        config_dir = Path(__file__).resolve().parents[1] / "configs"
        path = config_dir / "pipeline_v2_stage_a.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(config["stage"], "A")
        self.assertEqual(config["profiles"]["final"]["n_splits"], 5)
        self.assertEqual(len(config["profiles"]["final"]["seeds"]), 10)
        self.assertEqual(config["artifact_versions"], {"remoteclip": "v2", "xgboost": "v2"})
        self.assertNotIn("Outputs", config["paths"]["remoteclip_legacy_cohort"])
        self.assertNotIn("Outputs", config["paths"]["xgboost_legacy_cohort"])

        provenance = json.loads(
            (config_dir / "pipeline_v2_legacy_cohort_provenance.json").read_text(
                encoding="utf-8"
            )
        )
        with (config_dir / "pipeline_v2_remoteclip_legacy_cohort.csv").open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            remoteclip_rows = list(csv.DictReader(handle))
        remoteclip_keys = [
            {
                "legacy_event_id": row["legacy_event_id"],
                "sample_id": row["sample_id"],
                "event_year": int(row["event_year"]),
            }
            for row in remoteclip_rows
        ]
        self.assertEqual(
            stable_json_hash(remoteclip_keys),
            provenance["remoteclip"]["frozen_rows_sha256"],
        )
        self.assertEqual(
            stable_json_hash(sorted(row["legacy_event_id"] for row in remoteclip_keys)),
            provenance["remoteclip"]["artifact_record_ids_sha256"],
        )

        with (config_dir / "pipeline_v2_xgboost_legacy_cohort.csv").open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            xgboost_rows = list(csv.DictReader(handle))
        xgboost_keys = [
            {
                "sample_id": row["sample_id"],
                "event_year": int(row["event_year"]),
                "legacy_label": row["legacy_label"],
            }
            for row in xgboost_rows
        ]
        self.assertEqual(
            stable_json_hash(xgboost_keys),
            provenance["xgboost"]["frozen_rows_sha256"],
        )


if __name__ == "__main__":
    unittest.main()


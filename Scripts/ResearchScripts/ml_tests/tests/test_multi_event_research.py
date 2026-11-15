from __future__ import annotations

import unittest

from ml_tests.multi_event_research import (
    dominant_year_ranking,
    paired_model_bootstrap,
    point_taxonomy,
    validate_fixed_small_event_rules,
)


def event(
    sample_id: str,
    year: int,
    *,
    status: str,
    human_label: str | None = None,
    hierarchy: str | None = None,
    area: float = 10.0,
    pixels: int = 250,
    loss_percent: float = 2.0,
    remoteclip: str = "Wildfire",
    xgboost: str = "Wildfire",
    dominant: bool = False,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "event_id": f"{sample_id}-{year}",
        "event_year": year,
        "review_status": status,
        "human_label": human_label,
        "review_hierarchy_class": hierarchy,
        "event_loss_area_ha": area,
        "event_hansen_pixels": pixels,
        "hansen_loss_aoi_percent": loss_percent,
        "remoteclip_label": remoteclip,
        "xgboost_label": xgboost,
        "remoteclip_probability_wildfire": 0.9 if remoteclip == "Wildfire" else 0.1,
        "xgboost_probability_wildfire": 0.9 if xgboost == "Wildfire" else 0.1,
        "event_is_dominant": dominant,
        "usable_pre_download_count": 2,
        "usable_post_download_count": 2,
        "download_count": 4,
        "bad_quality_download_count": 0,
        "excluded_download_count": 0,
    }


class PairedModelBootstrapTests(unittest.TestCase):
    def test_remoteclip_delta_is_grouped_by_sample(self) -> None:
        rows = [
            event(
                "sample-a",
                2020,
                status="VALID",
                human_label="Logging",
                remoteclip="Logging",
                xgboost="Wildfire",
            ),
            event(
                "sample-a",
                2021,
                status="VALID",
                human_label="Logging",
                remoteclip="Logging",
                xgboost="Wildfire",
            ),
            event(
                "sample-b",
                2022,
                status="VALID",
                human_label="Wildfire",
                remoteclip="Wildfire",
                xgboost="Wildfire",
            ),
        ]

        result = paired_model_bootstrap(rows, bootstrap_runs=40, seed=7)
        accuracy = next(row for row in result["metric_rows"] if row["metric"] == "accuracy")

        self.assertEqual(result["summary"]["points"], 2)
        self.assertEqual(result["summary"]["bootstrap_unit"], "sample_id")
        self.assertGreater(accuracy["point_delta"], 0.0)
        self.assertGreater(accuracy["probability_delta_gt_zero"], 0.5)


class FrozenThresholdValidationTests(unittest.TestCase):
    def test_baseline_points_are_not_reused_for_validation(self) -> None:
        rows = [
            event(
                "baseline",
                2020,
                status="REJECTED",
                hierarchy="EVENT_TOO_SMALL",
                area=1.0,
                pixels=20,
                loss_percent=0.2,
            ),
            event(
                "new-small",
                2021,
                status="REJECTED",
                hierarchy="EVENT_TOO_SMALL",
                area=2.0,
                pixels=40,
                loss_percent=0.4,
            ),
            event(
                "new-valid",
                2022,
                status="VALID",
                human_label="Wildfire",
                hierarchy="VALID_WILDFIRE",
                area=20.0,
                pixels=500,
                loss_percent=5.0,
            ),
        ]

        result = validate_fixed_small_event_rules(rows, baseline_sample_ids={"baseline"})
        combined = next(row for row in result["metrics"] if row["feature"] == "combined_or")

        self.assertEqual(result["summary"]["validation_points"], 2)
        self.assertEqual(result["summary"]["validation_events"], 2)
        self.assertEqual(combined["true_positive"], 1)
        self.assertEqual(combined["false_positive"], 0)


class PointTaxonomyTests(unittest.TestCase):
    def test_taxonomy_separates_tails_repeats_and_class_changes(self) -> None:
        rows = [
            event("tail", 2020, status="VALID", human_label="Wildfire"),
            event("tail", 2021, status="REJECTED", hierarchy="YEAR_MISMATCH"),
            event("repeat", 2019, status="VALID", human_label="Wildfire"),
            event("repeat", 2022, status="VALID", human_label="Wildfire"),
            event("change", 2018, status="VALID", human_label="Logging"),
            event("change", 2023, status="VALID", human_label="Wildfire"),
        ]

        result = point_taxonomy(rows)
        by_id = {row["sample_id"]: row["point_type"] for row in result["point_rows"]}

        self.assertEqual(by_id["tail"], "SINGLE_PHYSICAL_EVENT_WITH_TEMPORAL_TAILS")
        self.assertEqual(by_id["repeat"], "REPEATED_SAME_CLASS")
        self.assertEqual(by_id["change"], "TRUE_CLASS_CHANGE")


class DominantYearRankingTests(unittest.TestCase):
    def test_all_valid_years_are_accepted_as_targets(self) -> None:
        rows = [
            event("point", 2020, status="VALID", human_label="Logging", area=5.0),
            event(
                "point",
                2021,
                status="REJECTED",
                hierarchy="YEAR_MISMATCH",
                area=2.0,
            ),
            event(
                "point",
                2022,
                status="VALID",
                human_label="Wildfire",
                area=30.0,
                dominant=True,
            ),
        ]

        result = dominant_year_ranking(rows, seed=11)
        max_area = next(
            row
            for row in result["point_predictions"]
            if row["method"] == "MAX_HANSEN_AREA"
        )

        self.assertEqual(max_area["target_years"], [2020, 2022])
        self.assertEqual(max_area["selected_year"], 2022)
        self.assertTrue(max_area["hit_at_1"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_SCRIPTS_ROOT = PROJECT_ROOT / "Scripts" / "ResearchScripts"
if str(RESEARCH_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_SCRIPTS_ROOT))

from ml_tests.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from ml_tests.pipeline import ForestMLPipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reproducible forest ML experiments.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--experiment", choices=("baseline", "all"), default="baseline")
    parser.add_argument("--profile", choices=("quick", "final"), default="quick")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, profile=args.profile)
    pipeline = ForestMLPipeline(config)
    print(f"Run directory: {pipeline.run_dir}")

    plan = pipeline.plan_event_backfill()
    print(f"Event-link dry-run: {plan['counts']}")
    manifest, audit = pipeline.build_manifest(backfill_plan=plan)
    print(
        "Dataset: "
        f"eligible_images={audit['eligible_rows']} "
        f"samples={audit['eligible_samples']} events={audit['eligible_events']}"
    )
    embeddings, cache_report = pipeline.compute_embeddings(manifest)
    print(f"Embedding cache: {cache_report['counts']}")
    result = pipeline.run_baseline(manifest, embeddings)
    print(
        "Baseline: "
        f"metric_rows={len(result['baseline_metrics'])} "
        f"review_rows={len(result['nomanual_review_queue'])}"
    )


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_SCRIPTS_ROOT = PROJECT_ROOT / "Scripts" / "ResearchScripts"
if str(RESEARCH_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_SCRIPTS_ROOT))

from ml_tests.stage_a import DEFAULT_STAGE_A_CONFIG, run_stage_a  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete Pipeline v2 Stage A (T00-T05)."
    )
    parser.add_argument("--config", default=str(DEFAULT_STAGE_A_CONFIG))
    parser.add_argument("--profile", choices=("quick", "final"), default="final")
    parser.add_argument(
        "--overwrite-artifacts",
        action="store_true",
        help="Replace existing v2 artifacts only when intentionally rerunning Stage A.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_stage_a(
        config_path=args.config,
        profile=args.profile,
        overwrite_artifacts=args.overwrite_artifacts,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


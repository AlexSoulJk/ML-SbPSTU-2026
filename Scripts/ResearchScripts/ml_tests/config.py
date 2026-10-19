from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "Scripts"

if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from ResearchTool.backend.config import load_dotenv  # noqa: E402


DEFAULT_CONFIG_PATH = PACKAGE_DIR / "configs" / "forest_ml_v1.json"


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExperimentConfig:
    path: Path
    data: dict[str, Any]
    profile_name: str

    @property
    def profile(self) -> dict[str, Any]:
        profiles = self.data.get("profiles", {})
        if self.profile_name not in profiles:
            raise KeyError(f"Unknown profile: {self.profile_name}")
        return dict(profiles[self.profile_name])

    def path_value(self, key: str) -> Path:
        raw = self.data.get("paths", {}).get(key)
        if not raw:
            raise KeyError(f"Missing paths.{key} in {self.path}")
        path = Path(os.path.expandvars(str(raw))).expanduser()
        return path if path.is_absolute() else PROJECT_ROOT / path

    def checkpoint_path(self) -> Path:
        embedding = self.data.get("embedding", {})
        env_name = str(embedding.get("checkpoint_env") or "REMOTECLIP_CHECKPOINT")
        raw = os.environ.get(env_name) or self.data.get("paths", {}).get("checkpoint")
        if not raw:
            raise RuntimeError(
                f"RemoteCLIP checkpoint is not configured. Set {env_name} or paths.checkpoint."
            )
        path = Path(os.path.expandvars(str(raw))).expanduser()
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def fingerprint(self) -> str:
        return stable_json_hash({"config": self.data, "profile": self.profile_name})


def load_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    profile: str = "quick",
) -> ExperimentConfig:
    load_dotenv()
    path = Path(config_path)
    if not path.is_absolute():
        candidate = PROJECT_ROOT / path
        path = candidate if candidate.exists() else path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Experiment config not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if int(data.get("version") or 0) != 1:
        raise ValueError(f"Unsupported config version in {path}: {data.get('version')}")
    return ExperimentConfig(path=path.resolve(), data=data, profile_name=profile)


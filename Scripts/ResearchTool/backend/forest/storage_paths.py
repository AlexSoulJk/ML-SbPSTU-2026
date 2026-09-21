from __future__ import annotations

from pathlib import Path, PurePosixPath

from ..config import FOREST_CACHE_DIR, PROJECT_ROOT


def _normalize_relative(path: Path | PurePosixPath | str) -> str:
    return PurePosixPath(*Path(path).parts).as_posix()


def to_storage_path(path: str | Path | None) -> str | None:
    if path is None:
        return None

    raw = Path(path)
    if not raw.is_absolute():
        return _normalize_relative(raw)

    project_root = PROJECT_ROOT.resolve()
    resolved = raw.resolve(strict=False)
    try:
        return _normalize_relative(resolved.relative_to(project_root))
    except ValueError:
        return str(raw)


def resolve_storage_path(stored_path: str | Path) -> Path:
    path = Path(stored_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def forest_cache_url(stored_or_absolute_path: str | Path) -> str | None:
    path = resolve_storage_path(stored_or_absolute_path)
    try:
        relative = path.resolve(strict=False).relative_to(FOREST_CACHE_DIR.resolve(strict=False))
    except ValueError:
        return None
    return "/cache/forest/" + "/".join(relative.parts)

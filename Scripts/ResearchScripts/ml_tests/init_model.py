from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_device(requested: str = "auto"):
    import torch

    normalized = requested.lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {requested}")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(normalized)


def load_remoteclip_model(
    *,
    model_name: str,
    checkpoint_path: str | Path,
    device: str = "auto",
) -> tuple[Any, Any, Any]:
    try:
        import open_clip
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "RemoteCLIP dependencies are missing. Install ml_tests/requirements.txt."
        ) from exc

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"RemoteCLIP checkpoint not found: {checkpoint_path}")

    resolved_device = resolve_device(device)
    model, _, preprocess = open_clip.create_model_and_transforms(model_name)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if isinstance(checkpoint, dict) and checkpoint:
        first_key = next(iter(checkpoint))
        if first_key.startswith("module."):
            checkpoint = {key.removeprefix("module."): value for key, value in checkpoint.items()}
    model.load_state_dict(checkpoint)
    model = model.to(resolved_device).eval()
    return model, preprocess, resolved_device

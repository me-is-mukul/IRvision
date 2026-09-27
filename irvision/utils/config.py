"""Load ``config/config.yaml`` and resolve repository-relative paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Return the config as a plain dict.

    Every entry under ``paths`` is converted to an absolute ``Path`` so callers
    never need to care about the current working directory.
    """
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {key: resolve_path(value) for key, value in cfg.get("paths", {}).items()}
    return cfg


def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to the repository root (absolute paths pass through)."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_device(cfg: dict[str, Any]) -> str:
    """Pick the torch device named in ``project.device`` (``auto`` prefers CUDA)."""
    requested = cfg.get("project", {}).get("device", "auto")
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    from irvision.utils.log import get_logger

    get_logger(__name__).warning(
        "CUDA not available (torch %s): running on CPU, training is ~10x slower. "
        "If you have an NVIDIA GPU, a CPU-only torch wheel is probably installed; see README → Installation.",
        torch.__version__,
    )
    return "cpu"

"""Cut aligned scene arrays into fixed-size patches and assign train/val/test.

Split strategy ("spatial"): each scene is divided into horizontal stripes
(e.g. top 70% train, next 15% val, bottom 15% test) and patches are cut
*inside* each stripe. No patch crosses a stripe boundary, so no pixel is
shared between splits even when patches overlap (stride < size). This
prevents train/test leakage.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SPLITS = ("train", "val", "test")


@dataclass
class Patch:
    row: int
    col: int
    split: str
    invalid_fraction: float
    ir: np.ndarray        # (1, S, S) normalized IR
    ir_clahe: np.ndarray  # (1, S, S) normalized + CLAHE IR (model input)
    rgb: np.ndarray       # (3, S, S) normalized RGB (target)
    valid: np.ndarray     # (S, S) bool
    extra: dict[str, np.ndarray] = field(default_factory=dict)  # optional (1, S, S) channels, e.g. ir_abs


def patch_windows(height: int, width: int, size: int, stride: int) -> list[tuple[int, int]]:
    """Top-left (row, col) of every *complete* ``size``x``size`` window."""
    if height < size or width < size:
        return []
    rows = range(0, height - size + 1, stride)
    cols = range(0, width - size + 1, stride)
    return [(r, c) for r in rows for c in cols]


def split_windows(
    height: int, width: int, size: int, stride: int, fractions: Sequence[float]
) -> list[tuple[str, int, int]]:
    """``(split, row, col)`` for every window lying entirely inside one split stripe.

    Each stripe gets its own window grid starting at the stripe's top edge, so
    even a stripe only slightly taller than ``size`` yields patches.
    """
    if len(fractions) != 3 or not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"split fractions must be 3 values summing to 1, got {fractions}")
    bounds = np.cumsum([0.0, *fractions]) * height
    windows = []
    for name, start, end in zip(SPLITS, bounds[:-1], bounds[1:]):
        top, bottom = int(np.ceil(start)), int(np.floor(end))
        for row, col in patch_windows(bottom - top, width, size, stride):
            windows.append((name, top + row, col))
    return windows


def extract_patches(
    ir: np.ndarray,
    ir_clahe: np.ndarray,
    rgb: np.ndarray,
    valid: np.ndarray,
    size: int = 256,
    stride: int = 256,
    max_invalid_fraction: float = 0.05,
    fractions: Sequence[float] = (0.7, 0.15, 0.15),
    force_split: str | None = None,
    extra: dict[str, np.ndarray] | None = None,
) -> Iterator[Patch]:
    """Yield patches that pass the validity filter.

    ``ir``/``ir_clahe``/``valid`` are (H, W); ``rgb`` is (3, H, W).
    ``force_split`` sends every patch of this scene to one split (used for
    whole held-out test scenes). ``extra`` maps names to additional (H, W)
    channels stored alongside (e.g. ``{"ir_abs": ...}``).
    """
    extra = extra or {}
    height, width = valid.shape
    if force_split:
        windows = [(force_split, r, c) for r, c in patch_windows(height, width, size, stride)]
    else:
        windows = split_windows(height, width, size, stride, fractions)

    for split, row, col in windows:
        window = np.s_[row : row + size, col : col + size]
        patch_valid = valid[window]
        invalid_fraction = 1.0 - float(patch_valid.mean())
        if invalid_fraction > max_invalid_fraction:
            continue
        yield Patch(
            row=row,
            col=col,
            split=split,
            invalid_fraction=invalid_fraction,
            ir=ir[window][None].copy(),
            ir_clahe=ir_clahe[window][None].copy(),
            rgb=rgb[(slice(None), *window)].copy(),
            valid=patch_valid.copy(),
            extra={name: arr[window][None].copy() for name, arr in extra.items()},
        )


def save_patch(path: str | Path, patch: Patch) -> None:
    """Save a patch as compressed ``.npz`` (float16 images, bool mask)."""
    np.savez_compressed(
        path,
        ir=patch.ir.astype(np.float16),
        ir_clahe=patch.ir_clahe.astype(np.float16),
        rgb=patch.rgb.astype(np.float16),
        valid=patch.valid,
        # float channels -> float16; integer channels (e.g. land-cover labels) keep their dtype
        **{name: arr.astype(np.float16) if np.issubdtype(arr.dtype, np.floating) else arr
           for name, arr in patch.extra.items()},
    )

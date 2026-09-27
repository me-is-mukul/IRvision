"""Contrast enhancement for single-channel IR images (CLAHE)."""

from __future__ import annotations

import cv2
import numpy as np


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Apply CLAHE to a float image in ``[0, 1]``; returns float32 in ``[0, 1]``.

    CLAHE runs in 16-bit so subtle thermal differences are not lost to 8-bit
    quantization. Invalid pixels are filled with the valid median beforehand
    (so they don't distort local histograms) and set back to 0 afterwards.
    """
    if image.ndim != 2:
        raise ValueError(f"apply_clahe expects a 2-D image, got shape {image.shape}")

    img = np.clip(np.nan_to_num(image.astype(np.float32)), 0.0, 1.0)
    if valid is not None and valid.any():
        img = np.where(valid, img, np.median(img[valid]))

    img16 = np.round(img * 65535).astype(np.uint16)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    out = clahe.apply(img16).astype(np.float32) / 65535.0

    if valid is not None:
        out = np.where(valid, out, 0.0).astype(np.float32)
    return out

"""Run a model on images of any size by blending overlapping tiles.

Used by the U-Net colorizer and the land-cover segmenter. Tiles are blended
with a feathered weight (ramping up over ``overlap`` px from each tile edge),
so no seams appear where tiles meet.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def tile_starts(length: int, size: int, step: int) -> list[int]:
    """Tile start positions covering [0, length), the last tile flush with the end."""
    if length <= size:
        return [0]
    starts = list(range(0, length - size, step))
    return starts + [length - size]


def tiled_apply(
    fn: Callable[[np.ndarray], np.ndarray],
    image: np.ndarray,
    out_channels: int,
    tile_size: int = 512,
    overlap: int = 64,
    scale: int = 1,
) -> np.ndarray:
    """Apply ``fn`` (C_in, h, w) -> (C_out, h*scale, w*scale) to ``image`` (C_in, H, W) tile by tile.

    ``scale`` > 1 is for upsampling models (super-resolution). Tiles are chosen in
    input coordinates and blended in output coordinates. Images no larger than
    ``tile_size`` are processed in a single call.
    """
    _, h, w = image.shape
    if max(h, w) <= tile_size:
        return fn(image)

    step = tile_size - overlap
    out = np.zeros((out_channels, h * scale, w * scale), np.float32)
    weight = np.zeros((h * scale, w * scale), np.float32)
    size_out, overlap_out = tile_size * scale, overlap * scale
    ramp = np.minimum(1.0, (np.arange(size_out) + 1) / (overlap_out + 1)).astype(np.float32)
    feather = np.minimum(ramp, ramp[::-1])
    for top in tile_starts(h, tile_size, step):
        for left in tile_starts(w, tile_size, step):
            tile = image[:, top : top + tile_size, left : left + tile_size]
            th, tw = tile.shape[1] * scale, tile.shape[2] * scale
            t0, l0 = top * scale, left * scale
            wgt = np.outer(feather[:th], feather[:tw])
            out[:, t0 : t0 + th, l0 : l0 + tw] += fn(tile) * wgt
            weight[t0 : t0 + th, l0 : l0 + tw] += wgt
    return out / np.maximum(weight, 1e-6)

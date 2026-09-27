"""Non-learned IR -> RGB baselines. The U-Net must beat these to be worth it.

Every colorizer has the same interface, which the U-Net will share later:

    colorizer.colorize(ir)  # ir: (H, W) float in [0, 1] (normalized + CLAHE)
                            # -> (3, H, W) float32 RGB in [0, 1]

* ``MeanColorColorizer`` — ignores the IR and predicts the average train colour
                           everywhere. The *floor*: a model that doesn't beat it
                           has learned nothing from the IR.
* ``GrayColorizer``     — IR copied into R, G and B.
* ``ColormapColorizer`` — classic pseudo-colour (PLAN.md §9), e.g. inferno.
                           Readable for humans, but not meant to match true colour.
* ``LUTColorizer``      — "average true colour for each IR brightness", a
                           256-entry lookup table fitted on the train split.
                           The strongest possible *per-pixel* mapping; anything
                           better has to use spatial context (which U-Net does).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from matplotlib import colormaps


class MeanColorColorizer:
    name = "mean_color"

    def __init__(self, color: np.ndarray):
        self.color = np.asarray(color, dtype=np.float32).reshape(3)

    def colorize(self, ir: np.ndarray) -> np.ndarray:
        return np.broadcast_to(self.color[:, None, None], (3, *ir.shape)).copy()

    @classmethod
    def fit(cls, samples: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> "MeanColorColorizer":
        """Mean RGB over valid pixels of ``(ir, rgb, valid)`` samples (train split only!)."""
        total, count = np.zeros(3), 0
        for _, rgb, valid in samples:
            total += rgb[:, valid].sum(axis=1)
            count += int(valid.sum())
        if count == 0:
            raise ValueError("No valid pixels to fit the mean colour")
        return cls(total / count)


class GrayColorizer:
    name = "gray"

    def colorize(self, ir: np.ndarray) -> np.ndarray:
        return np.repeat(np.clip(ir, 0, 1)[None], 3, axis=0).astype(np.float32)


class ColormapColorizer:
    def __init__(self, cmap: str = "inferno"):
        self.cmap = colormaps[cmap]
        self.name = f"colormap_{cmap}"

    def colorize(self, ir: np.ndarray) -> np.ndarray:
        rgba = self.cmap(np.clip(ir, 0, 1))          # (H, W, 4)
        return np.moveaxis(rgba[..., :3], -1, 0).astype(np.float32)


class LUTColorizer:
    name = "lut"

    def __init__(self, lut: np.ndarray):
        if lut.ndim != 2 or lut.shape[1] != 3:
            raise ValueError(f"LUT must have shape (bins, 3), got {lut.shape}")
        self.lut = lut.astype(np.float32)

    @property
    def bins(self) -> int:
        return len(self.lut)

    def colorize(self, ir: np.ndarray) -> np.ndarray:
        idx = np.clip((np.clip(ir, 0, 1) * self.bins).astype(np.int64), 0, self.bins - 1)
        return np.moveaxis(self.lut[idx], -1, 0)

    @classmethod
    def fit(cls, samples: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]], bins: int = 256) -> "LUTColorizer":
        """Fit from ``(ir (H,W), rgb (3,H,W), valid (H,W))`` samples (train split only!).

        Bins with no training pixels are filled by linear interpolation.
        """
        sums, counts = np.zeros((bins, 3)), np.zeros(bins)
        for ir, rgb, valid in samples:
            idx = np.clip((ir[valid] * bins).astype(np.int64), 0, bins - 1)
            counts += np.bincount(idx, minlength=bins)
            for c in range(3):
                sums[:, c] += np.bincount(idx, weights=rgb[c][valid], minlength=bins)
        filled = counts > 0
        if not filled.any():
            raise ValueError("No valid pixels to fit the LUT")
        centres = np.arange(bins)
        lut = np.stack([np.interp(centres, centres[filled], sums[filled, c] / counts[filled]) for c in range(3)], 1)
        return cls(lut)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.lut)

    @classmethod
    def load(cls, path: str | Path) -> "LUTColorizer":
        return cls(np.load(path))

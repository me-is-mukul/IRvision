"""Geometric augmentation applied identically to input, target and mask.

Only flips and 90° rotations: they are exact (no interpolation) and a satellite
image has no preferred orientation. Colour/brightness jitter is deliberately not
used because the target colours are what the model must learn.
"""

from __future__ import annotations

import numpy as np


class RandomFlipRotate:
    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def __call__(self, x: np.ndarray, y: np.ndarray, valid: np.ndarray):
        k = int(self.rng.integers(4))
        flip = bool(self.rng.integers(2))
        out = []
        for a in (x, y, valid):          # all are (C, H, W)
            a = np.rot90(a, k, axes=(-2, -1))
            out.append(a[..., ::-1] if flip else a)
        return tuple(out)

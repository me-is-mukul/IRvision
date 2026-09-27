"""Land-cover classes used for semantic validation, derived from ESA WorldCover.

ESA WorldCover 2021 v200 (10 m, 11 classes) is merged into 5 classes that are
distinguishable in 30 m RGB imagery. 255 = ignore (no data).
"""

from __future__ import annotations

import numpy as np

IGNORE = 255

CLASSES = ["tree cover", "low vegetation / crops", "built-up", "bare / sparse", "water"]

# display colours (RGB 0-255) for each class
PALETTE = np.array([
    [0, 100, 0],        # tree cover: dark green
    [170, 210, 90],     # low vegetation / crops: light green
    [220, 40, 40],      # built-up: red
    [200, 170, 120],    # bare / sparse: sand
    [30, 90, 220],      # water: blue
], dtype=np.uint8)

# ESA WorldCover code -> our class index
WORLDCOVER_TO_CLASS = {
    10: 0,   # tree cover
    95: 0,   # mangroves
    20: 1,   # shrubland
    30: 1,   # grassland
    40: 1,   # cropland
    90: 1,   # herbaceous wetland
    100: 1,  # moss and lichen
    50: 2,   # built-up
    60: 3,   # bare / sparse vegetation
    70: 3,   # snow and ice
    80: 4,   # permanent water bodies
}


def remap_worldcover(codes: np.ndarray) -> np.ndarray:
    """WorldCover codes (uint8) -> class indices; unknown / 0 (no data) -> IGNORE."""
    lut = np.full(256, IGNORE, dtype=np.uint8)
    for code, cls in WORLDCOVER_TO_CLASS.items():
        lut[code] = cls
    return lut[codes.astype(np.uint8)]


def colorize_labels(labels: np.ndarray) -> np.ndarray:
    """(H, W) class indices -> (H, W, 3) uint8 image; IGNORE is shown black."""
    out = np.zeros((*labels.shape, 3), dtype=np.uint8)
    known = labels != IGNORE
    out[known] = PALETTE[labels[known]]
    return out

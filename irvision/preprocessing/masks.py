"""Decode the Landsat Collection-2 ``QA_PIXEL`` band into validity masks.

Bit layout (USGS Landsat 8-9 C2 L2 Product Guide, Table 6-3):
    0 fill, 1 dilated cloud, 2 cirrus, 3 cloud, 4 cloud shadow,
    5 snow, 6 clear, 7 water
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

QA_BITS = {
    "fill": 0,
    "dilated_cloud": 1,
    "cirrus": 2,
    "cloud": 3,
    "cloud_shadow": 4,
    "snow": 5,
    "clear": 6,
    "water": 7,
}


def qa_flag(qa: np.ndarray, flag: str) -> np.ndarray:
    """Boolean array that is True where ``flag`` is set in ``qa``."""
    if flag not in QA_BITS:
        raise ValueError(f"Unknown QA flag {flag!r}; expected one of {sorted(QA_BITS)}")
    return (qa.astype(np.uint16) >> QA_BITS[flag]) & 1 == 1


def qa_invalid_mask(qa: np.ndarray, flags: Iterable[str]) -> np.ndarray:
    """True where *any* of ``flags`` is set, i.e. the pixel should not be used."""
    invalid = np.zeros(qa.shape, dtype=bool)
    for flag in flags:
        invalid |= qa_flag(qa, flag)
    return invalid

"""Scale physical values (Kelvin, reflectance) to float32 in [0, 1].

Invalid pixels (``valid == False``) are ignored when computing statistics and
are set to 0 in the output. The statistics used are returned so they can be
stored with the dataset and re-applied at inference time.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def range_normalize(x: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Linearly map ``[vmin, vmax]`` to ``[0, 1]`` and clip."""
    if vmax <= vmin:
        raise ValueError(f"vmax ({vmax}) must be greater than vmin ({vmin})")
    out = (x.astype(np.float32) - vmin) / (vmax - vmin)
    return np.clip(out, 0.0, 1.0)


def percentile_normalize(
    x: np.ndarray, low: float = 2, high: float = 98, valid: np.ndarray | None = None
) -> tuple[np.ndarray, tuple[float, float]]:
    """Robust min/max stretch using the ``low``/``high`` percentiles of valid pixels.

    Returns ``(normalized, (vmin, vmax))``. A constant image maps to all zeros.
    """
    values = x[valid] if valid is not None else x[np.isfinite(x)]
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("No valid pixels to compute normalization statistics from")
    vmin, vmax = (float(v) for v in np.percentile(values, [low, high]))
    if vmax <= vmin:  # flat image: avoid division by zero
        return np.zeros_like(x, dtype=np.float32), (vmin, vmin)
    return range_normalize(x, vmin, vmax), (vmin, vmax)


def normalize(
    x: np.ndarray, cfg: dict[str, Any], valid: np.ndarray | None = None
) -> tuple[np.ndarray, dict[str, Any]]:
    """Normalize using a config block such as ``preprocessing.ir_normalization``.

    Works for ``(H, W)`` or ``(C, H, W)`` arrays (statistics are shared across
    channels). Returns ``(normalized, stats)`` where ``stats`` records what was
    applied.
    """
    method = cfg["method"]
    if method == "percentile":
        pixel_valid = None if valid is None else np.broadcast_to(valid, x.shape)
        out, (vmin, vmax) = percentile_normalize(x, cfg["low"], cfg["high"], pixel_valid)
    elif method == "range":
        vmin, vmax = float(cfg["min"]), float(cfg["max"])
        out = range_normalize(x, vmin, vmax)
    else:
        raise ValueError(f"Unknown normalization method {method!r} (use 'percentile' or 'range')")

    out = np.nan_to_num(out, nan=0.0)
    if valid is not None:
        out = np.where(valid, out, 0.0).astype(np.float32)
    return out.astype(np.float32), {"method": method, "vmin": vmin, "vmax": vmax}

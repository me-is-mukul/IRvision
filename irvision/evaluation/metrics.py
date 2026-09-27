"""Image-quality metrics (PSNR, SSIM) and timing.

All images are float arrays in [0, 1], shape (C, H, W) or (H, W). An optional
``valid`` mask (H, W) restricts the metric to usable pixels (no cloud/fill);
every reported number in this project uses the mask when one is available.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
from skimage.metrics import structural_similarity


def _as_chw(x: np.ndarray) -> np.ndarray:
    return x[None] if x.ndim == 2 else x


def psnr(pred: np.ndarray, target: np.ndarray, valid: np.ndarray | None = None, data_range: float = 1.0) -> float:
    """Peak signal-to-noise ratio in dB (higher is better; ``inf`` if identical)."""
    pred, target = _as_chw(pred).astype(np.float64), _as_chw(target).astype(np.float64)
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs target {target.shape}")
    sq_err = (pred - target) ** 2
    mse = sq_err[:, valid].mean() if valid is not None else sq_err.mean()
    return float("inf") if mse == 0 else float(10 * np.log10(data_range**2 / mse))


def ssim(pred: np.ndarray, target: np.ndarray, valid: np.ndarray | None = None, data_range: float = 1.0) -> float:
    """Mean structural similarity over channels (1.0 = identical).

    Uses scikit-image's SSIM map (7x7 window) averaged over valid pixels.
    """
    pred, target = _as_chw(pred).astype(np.float64), _as_chw(target).astype(np.float64)
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs target {target.shape}")
    _, ssim_map = structural_similarity(pred, target, channel_axis=0, data_range=data_range, full=True)
    ssim_map = ssim_map.mean(axis=0)
    return float(ssim_map[valid].mean() if valid is not None else ssim_map.mean())


def time_function(fn: Callable[[], object], repeats: int = 20, warmup: int = 3) -> dict[str, float]:
    """Wall-clock timing of ``fn()`` in milliseconds: mean, std, median.

    For GPU code, ``fn`` must synchronize (e.g. ``torch.cuda.synchronize()``)
    so the timing includes the actual compute.
    """
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    t = np.array(times)
    return {"mean_ms": float(t.mean()), "std_ms": float(t.std()), "median_ms": float(np.median(t))}

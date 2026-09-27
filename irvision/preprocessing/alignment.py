"""Co-registration of bands and a quantitative IR<->RGB alignment check.

Two separate jobs live here:

1. ``read_on_grid`` — put any raster onto a reference pixel grid (same CRS,
   transform and shape), reprojecting/resampling only when needed.
2. ``verify_alignment`` — *measure* the residual shift between the IR and RGB
   images with phase correlation on gradient images (edges are the only thing
   thermal and visible bands reliably share). A positive control (an injected,
   known shift) proves the measurement actually works on that scene.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject


@dataclass(frozen=True)
class GridSpec:
    """A raster pixel grid: coordinate system + pixel-to-map transform + size."""

    crs: CRS
    transform: Affine
    height: int
    width: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @classmethod
    def from_file(cls, path: str | Path) -> "GridSpec":
        with rasterio.open(path) as src:
            return cls(src.crs, src.transform, src.height, src.width)


def read_on_grid(
    path: str | Path, grid: GridSpec, resampling: str = "bilinear", fill_value: int = 0
) -> tuple[np.ndarray, bool]:
    """Read band 1 of ``path`` on ``grid``. Returns ``(array, was_resampled)``.

    Pixels with no source data are set to ``fill_value`` (Landsat uses 0 for
    SR/ST bands and 1 — the fill bit — for QA_PIXEL).
    """
    with rasterio.open(path) as src:
        if src.crs == grid.crs and src.transform == grid.transform and (src.height, src.width) == grid.shape:
            return src.read(1), False

        dest = np.full(grid.shape, fill_value, dtype=src.dtypes[0])
        reproject(
            source=rasterio.band(src, 1),
            destination=dest,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata if src.nodata is not None else fill_value,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            dst_nodata=fill_value,
            resampling=Resampling[resampling],
        )
        return dest, True


# ---------------------------------------------------------------------------
# Alignment verification
# ---------------------------------------------------------------------------


@dataclass
class AlignmentReport:
    status: str                      # "pass" | "fail" | "inconclusive"
    shift_px: tuple[float, float]    # measured (dx, dy) of RGB relative to IR
    shift_magnitude: float
    response: float                  # phase-correlation peak strength (0..1)
    control_expected: tuple[float, float]
    control_measured: tuple[float, float]
    control_ok: bool
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


def _gradient_image(image: np.ndarray, valid: np.ndarray, blur_sigma: float) -> np.ndarray:
    """Standardized gradient magnitude, zeroed near invalid pixels."""
    img = np.where(valid, np.nan_to_num(image), np.median(image[valid])).astype(np.float32)
    if blur_sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur_sigma)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx**2 + gy**2)
    # mask boundaries create strong artificial edges; suppress them
    eroded = cv2.erode(valid.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    grad[~eroded] = 0.0
    std = grad[eroded].std() if eroded.any() else 0.0
    return ((grad - grad[eroded].mean()) / std if std > 0 else grad) * eroded


def estimate_shift(
    reference: np.ndarray, moving: np.ndarray, valid: np.ndarray, blur_sigma: float = 1.5
) -> tuple[tuple[float, float], float]:
    """Estimate the (dx, dy) translation of ``moving`` relative to ``reference``.

    Returns ``((dx, dy), response)``: if ``moving`` equals ``reference`` shifted
    right by 3 px, ``dx ≈ +3``.
    """
    a = _gradient_image(reference, valid, blur_sigma)
    b = _gradient_image(moving, valid, blur_sigma)
    window = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(a.astype(np.float32), b.astype(np.float32), window)
    return (float(dx), float(dy)), float(response)


def verify_alignment(
    ir: np.ndarray,
    rgb: np.ndarray,
    valid: np.ndarray,
    max_shift_px: float = 1.0,
    min_response: float = 0.02,
    control_shift: tuple[int, int] = (7, -4),
) -> AlignmentReport:
    """Check that ``ir`` (H, W) and ``rgb`` (3, H, W) are spatially aligned.

    Measures the residual shift between the IR and the RGB luminance. Then
    re-measures after shifting the luminance by ``control_shift``; if that
    known shift is not recovered, the check is reported as "inconclusive"
    instead of silently passing.
    """
    luminance = np.nan_to_num(rgb).mean(axis=0)
    ir = np.nan_to_num(ir)

    (dx, dy), response = estimate_shift(ir, luminance, valid)
    magnitude = float(np.hypot(dx, dy))

    cdx, cdy = control_shift
    shifted = np.roll(luminance, shift=(cdy, cdx), axis=(0, 1))
    shifted_valid = valid & np.roll(valid, shift=(cdy, cdx), axis=(0, 1))
    (mdx, mdy), _ = estimate_shift(ir, shifted, shifted_valid)
    # the control is measured relative to the scene's own residual shift
    control_ok = abs((mdx - dx) - cdx) <= 1.0 and abs((mdy - dy) - cdy) <= 1.0

    if not control_ok:
        status, msg = "inconclusive", "positive control failed: the shift estimate is unreliable on this scene"
    elif response < min_response:
        status, msg = "inconclusive", f"weak correlation peak ({response:.3f} < {min_response})"
    elif magnitude > max_shift_px:
        status, msg = "fail", f"residual shift {magnitude:.2f}px exceeds {max_shift_px}px"
    else:
        status, msg = "pass", f"residual shift {magnitude:.2f}px <= {max_shift_px}px"

    return AlignmentReport(
        status=status,
        shift_px=(dx, dy),
        shift_magnitude=magnitude,
        response=response,
        control_expected=(float(cdx), float(cdy)),
        control_measured=(mdx - dx, mdy - dy),
        control_ok=control_ok,
        message=msg,
    )

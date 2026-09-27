"""Read user-supplied images (uploads, files) into numpy arrays.

Supported: GeoTIFF / TIFF (via rasterio, keeps 16-bit and nodata), PNG, JPEG
(via OpenCV, keeps 16-bit PNGs). Colour images are converted to one channel
for IR input, with a warning, since thermal data is single-channel.
"""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import rasterio

TIFF_SUFFIXES = (".tif", ".tiff")


def read_image(source: str | Path | bytes, filename: str | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Return ``(array, nodata_mask)``.

    ``array`` is (H, W) or (H, W, C) in the file's own dtype/scale;
    ``nodata_mask`` is True where the file declares no data (GeoTIFF nodata,
    or 0 in Landsat ``*_ST_B10`` / ``*_SR_B*`` products), else ``None``.
    """
    name = (filename or (str(source) if not isinstance(source, bytes) else "")).lower()
    data = source if isinstance(source, bytes) else Path(source).read_bytes()

    if name.endswith(TIFF_SUFFIXES):
        with rasterio.MemoryFile(data) as mem, mem.open() as src:
            arr = src.read()                                   # (C, H, W)
            nodata = src.nodata
        if nodata is None and any(tag in name for tag in ("_st_b10", "_sr_b")):
            nodata = 0                                         # Landsat C2 L2 fill value
        arr = arr[0] if arr.shape[0] == 1 else np.moveaxis(arr, 0, -1)
        mask = None
        if nodata is not None:
            hit = np.isnan(arr) if np.isnan(nodata) else arr == nodata   # NaN never equals NaN
            mask = hit if arr.ndim == 2 else np.all(hit, axis=-1)
        return arr, mask

    buf = np.frombuffer(data, np.uint8)
    arr = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError(f"Could not decode image {filename or source!r} (supported: TIFF, PNG, JPEG)")
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA if arr.shape[2] == 4 else cv2.COLOR_BGR2RGB)
    return arr, None


def to_single_channel(arr: np.ndarray) -> tuple[np.ndarray, str | None]:
    """(H, W, C) -> (H, W). Returns ``(image, warning or None)``."""
    if arr.ndim == 2:
        return arr, None
    if arr.ndim == 3 and arr.shape[2] == 1:
        return arr[..., 0], None
    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        rgb = arr[..., :3].astype(np.float32)
        if np.allclose(rgb[..., 0], rgb[..., 1]) and np.allclose(rgb[..., 1], rgb[..., 2]):
            return rgb[..., 0], None                           # grayscale saved as RGB
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        return gray, "Input has colour channels; converted to grayscale. IR input should be single-channel."
    raise ValueError(f"Unsupported image shape {arr.shape}")


def to_png_bytes(rgb_chw: np.ndarray) -> bytes:
    """(3, H, W) float in [0, 1] -> PNG bytes (8-bit)."""
    img = (np.clip(np.moveaxis(rgb_chw, 0, -1), 0, 1) * 255).round().astype(np.uint8)
    ok, png = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return io.BytesIO(png.tobytes()).getvalue()

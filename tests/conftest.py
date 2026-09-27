"""Shared fixtures: a small synthetic Landsat scene written as real GeoTIFFs."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from irvision.preprocessing.landsat import SR_OFFSET, SR_SCALE, ST_OFFSET, ST_SCALE

SCENE_ID = "LC09_L2SP_000000_20240101_20240102_02_T1"
SIZE = 128          # pixels at 30 m
CRS = "EPSG:32643"
ORIGIN = (500000.0, 1400000.0)

CLOUD_BLOCK = np.s_[20:30, 40:50]
FILL_COLS = 4       # left-most columns are fill (no data)


def textured_field(size: int, seed: int = 0, blur: float = 3.0) -> np.ndarray:
    """Smooth random texture in [0, 1] — has edges for alignment tests."""
    rng = np.random.default_rng(seed)
    field = cv2.GaussianBlur(rng.random((size, size)).astype(np.float32), (0, 0), blur)
    return (field - field.min()) / (field.max() - field.min())


def _write(path, data, transform):
    profile = dict(driver="GTiff", width=data.shape[1], height=data.shape[0], count=1,
                   dtype=data.dtype, crs=CRS, transform=transform, nodata=None)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


@pytest.fixture
def synthetic_scene_dir(tmp_path):
    """Scene folder: SR bands + QA at 30 m (128x128), ST_B10 at 60 m (64x64)."""
    base = textured_field(SIZE)
    t30 = from_origin(*ORIGIN, 30, 30)
    t60 = from_origin(*ORIGIN, 60, 60)

    for i, suffix in enumerate(("SR_B4", "SR_B3", "SR_B2")):
        reflectance = 0.05 + 0.2 * base * (1 - 0.1 * i)
        dn = np.round((reflectance - SR_OFFSET) / SR_SCALE).astype(np.uint16)
        dn[:, :FILL_COLS] = 0
        _write(tmp_path / f"{SCENE_ID}_{suffix}.TIF", dn, t30)

    kelvin = 290 + 30 * cv2.resize(base, (SIZE // 2, SIZE // 2), interpolation=cv2.INTER_AREA)
    st_dn = np.round((kelvin - ST_OFFSET) / ST_SCALE).astype(np.uint16)
    _write(tmp_path / f"{SCENE_ID}_ST_B10.TIF", st_dn, t60)

    qa = np.full((SIZE, SIZE), 1 << 6, dtype=np.uint16)       # clear
    qa[CLOUD_BLOCK] = (1 << 3) | (1 << 1)                     # cloud + dilated cloud
    qa[:, :FILL_COLS] = 1                                     # fill
    _write(tmp_path / f"{SCENE_ID}_QA_PIXEL.TIF", qa, t30)
    return tmp_path

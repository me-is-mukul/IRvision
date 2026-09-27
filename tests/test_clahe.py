import numpy as np
import pytest

from irvision.preprocessing.enhancement import apply_clahe
from tests.conftest import textured_field


def test_clahe_increases_contrast_and_keeps_range():
    low_contrast = 0.45 + 0.1 * textured_field(256)  # values in [0.45, 0.55]
    out = apply_clahe(low_contrast, clip_limit=2.0, tile_grid_size=8)
    assert out.shape == low_contrast.shape and out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0
    assert out.std() > 2 * low_contrast.std()


def test_clahe_invalid_pixels_zeroed():
    img = textured_field(64)
    valid = np.ones_like(img, dtype=bool)
    valid[:8] = False
    out = apply_clahe(img, valid=valid)
    assert (out[:8] == 0).all()
    assert out[8:].max() > 0


def test_clahe_rejects_multichannel():
    with pytest.raises(ValueError):
        apply_clahe(np.zeros((3, 8, 8)))

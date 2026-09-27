import numpy as np
import pytest

from irvision.preprocessing.normalization import normalize, percentile_normalize, range_normalize


def test_range_normalize_clips():
    out = range_normalize(np.array([-1.0, 0.0, 0.15, 0.3, 1.0]), 0.0, 0.3)
    np.testing.assert_allclose(out, [0, 0, 0.5, 1, 1], atol=1e-6)
    assert out.dtype == np.float32


def test_percentile_ignores_invalid_pixels():
    x = np.linspace(280, 320, 100).reshape(10, 10).astype(np.float32)
    valid = np.ones_like(x, dtype=bool)
    x[0, 0], valid[0, 0] = 10_000, False   # outlier in an invalid pixel
    out, (vmin, vmax) = percentile_normalize(x, 0, 100, valid)
    assert vmax < 321
    assert out[valid].min() == 0 and out[valid].max() == 1


def test_percentile_flat_image():
    out, _ = percentile_normalize(np.full((4, 4), 300.0))
    assert np.all(out == 0)


def test_normalize_multichannel_and_mask():
    x = np.random.default_rng(0).uniform(0, 0.3, (3, 8, 8)).astype(np.float32)
    x[:, 0, 0] = np.nan
    valid = np.ones((8, 8), bool)
    valid[0, 0] = False
    out, stats = normalize(x, {"method": "percentile", "low": 2, "high": 98}, valid)
    assert out.shape == (3, 8, 8) and out.dtype == np.float32
    assert np.isfinite(out).all()
    assert (out[:, 0, 0] == 0).all()
    assert 0 <= out.min() and out.max() <= 1
    assert stats["method"] == "percentile"


def test_normalize_unknown_method():
    with pytest.raises(ValueError):
        normalize(np.zeros((2, 2)), {"method": "zscore"})

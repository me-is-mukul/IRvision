import time

import numpy as np
import pytest

from irvision.evaluation.metrics import psnr, ssim, time_function
from tests.conftest import textured_field


def test_psnr_known_value():
    target = np.zeros((3, 16, 16))
    pred = target + 0.1                      # MSE = 0.01 -> 20 dB
    assert psnr(pred, target) == pytest.approx(20.0)


def test_psnr_identical_is_inf():
    x = np.random.default_rng(0).random((3, 8, 8))
    assert psnr(x, x) == float("inf")


def test_psnr_ignores_invalid_pixels():
    target = np.zeros((3, 16, 16))
    pred = target.copy()
    pred[:, :4] = 1.0                        # large error only in masked rows
    valid = np.ones((16, 16), bool)
    valid[:4] = False
    assert psnr(pred, target, valid) == float("inf")
    assert psnr(pred, target) < 10


def test_ssim_identical_and_degraded():
    img = np.stack([textured_field(64, seed=s) for s in range(3)])
    assert ssim(img, img) == pytest.approx(1.0)
    noisy = np.clip(img + np.random.default_rng(1).normal(0, 0.2, img.shape), 0, 1)
    assert ssim(noisy, img) < 0.8


def test_metrics_reject_shape_mismatch():
    with pytest.raises(ValueError):
        psnr(np.zeros((3, 8, 8)), np.zeros((1, 8, 8)))
    with pytest.raises(ValueError):
        ssim(np.zeros((3, 8, 8)), np.zeros((3, 8, 9)))


def test_time_function_measures_ms():
    t = time_function(lambda: time.sleep(0.01), repeats=3, warmup=0)
    assert 8 < t["mean_ms"] < 200

import numpy as np

from irvision.preprocessing.alignment import estimate_shift, verify_alignment
from tests.conftest import textured_field


def _pair(size=256):
    ir = textured_field(size, seed=1)
    rgb = np.stack([1 - ir, 0.8 * (1 - ir), 0.5 + 0.2 * ir])  # different "look", same edges
    return ir, rgb, np.ones((size, size), bool)


def test_estimate_shift_sign_convention():
    ir, _, valid = _pair()
    moved = np.roll(ir, shift=(-2, 3), axis=(0, 1))  # right by 3, up by 2
    (dx, dy), response = estimate_shift(ir, moved, valid)
    assert abs(dx - 3) < 0.5 and abs(dy + 2) < 0.5
    assert response > 0.1


def test_aligned_pair_passes():
    ir, rgb, valid = _pair()
    report = verify_alignment(ir, rgb, valid, max_shift_px=1.0)
    assert report.status == "pass", report.message
    assert report.control_ok
    assert report.shift_magnitude < 0.5


def test_misaligned_pair_fails():
    ir, rgb, valid = _pair()
    rgb_shifted = np.roll(rgb, shift=(0, 4), axis=(1, 2))
    report = verify_alignment(ir, rgb_shifted, valid, max_shift_px=1.0)
    assert report.status == "fail"
    assert abs(report.shift_px[0] - 4) < 0.5

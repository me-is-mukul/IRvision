import numpy as np
import pytest

from irvision.evaluation.evaluate import evaluate_colorizer, scene_of
from irvision.models.baseline import ColormapColorizer, GrayColorizer, LUTColorizer, MeanColorColorizer
from irvision.preprocessing.patches import Patch, save_patch
from tests.conftest import textured_field

IR = textured_field(32)


@pytest.mark.parametrize("colorizer", [MeanColorColorizer([0.3, 0.3, 0.2]), GrayColorizer(),ColormapColorizer("inferno"), LUTColorizer(np.zeros((256, 3)))])
def test_colorizers_output_contract(colorizer):
    out = colorizer.colorize(IR)
    assert out.shape == (3, 32, 32) and out.dtype == np.float32
    assert 0 <= out.min() and out.max() <= 1


def test_lut_learns_a_known_mapping():
    # ground truth: R = ir, G = 1 - ir, B = 0.5 (a per-pixel function the LUT can represent)
    rgb = np.stack([IR, 1 - IR, np.full_like(IR, 0.5)])
    lut = LUTColorizer.fit([(IR, rgb, np.ones_like(IR, bool))], bins=64)
    pred = lut.colorize(IR)
    assert np.abs(pred - rgb).max() < 1 / 64 + 1e-3   # error bounded by the bin width


def test_lut_ignores_invalid_and_fills_empty_bins():
    ir = np.array([[0.1, 0.9]], np.float32)
    rgb = np.stack([ir, ir, ir])
    valid = np.array([[True, True]])
    lut = LUTColorizer.fit([(ir, rgb, valid)], bins=10)
    assert np.isfinite(lut.lut).all()
    np.testing.assert_allclose(lut.colorize(np.array([[0.5]]))[:, 0, 0], 0.5, atol=1e-6)  # interpolated


def test_mean_color_fit_ignores_invalid():
    rgb = np.zeros((3, 4, 4), np.float32)
    rgb[:, :2] = 0.5
    valid = np.zeros((4, 4), bool)
    valid[:2] = True                         # only the 0.5 rows are valid
    mc = MeanColorColorizer.fit([(None, rgb, valid)])
    np.testing.assert_allclose(mc.colorize(IR), 0.5)


def test_lut_save_load(tmp_path):
    lut = LUTColorizer(np.random.default_rng(0).random((256, 3)))
    lut.save(tmp_path / "lut.npy")
    np.testing.assert_array_equal(LUTColorizer.load(tmp_path / "lut.npy").lut, lut.lut)


def test_evaluate_colorizer_on_patch_folder(tmp_path):
    rgb = np.stack([IR, IR, IR])
    patch = Patch(0, 0, "test", 0.0, IR[None], IR[None], rgb, np.ones_like(IR, bool))
    save_patch(tmp_path / "LC09_X_T1_r00000_c00000.npz", patch)
    df = evaluate_colorizer(GrayColorizer(), tmp_path)
    assert len(df) == 1 and df.loc[0, "scene_id"] == "LC09_X_T1"
    assert df.loc[0, "psnr"] > 40          # gray == target up to float16 storage error
    assert df.loc[0, "ssim"] > 0.99


def test_scene_of():
    assert scene_of("LC09_L2SP_144051_20240318_02_T1_r01741_c00512") == "LC09_L2SP_144051_20240318_02_T1"

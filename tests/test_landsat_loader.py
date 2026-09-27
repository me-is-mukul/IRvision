import numpy as np
import pytest

from irvision.preprocessing.landsat import find_band_files, load_scene
from tests.conftest import CLOUD_BLOCK, FILL_COLS, SCENE_ID, SIZE


def test_find_band_files(synthetic_scene_dir):
    files = find_band_files(synthetic_scene_dir)
    assert set(files) == {"red", "green", "blue", "ir", "qa"}
    assert files["ir"].name.endswith("_ST_B10.TIF")


def test_missing_band_raises(synthetic_scene_dir):
    (synthetic_scene_dir / f"{SCENE_ID}_ST_B10.TIF").unlink()
    with pytest.raises(FileNotFoundError, match="ST_B10"):
        find_band_files(synthetic_scene_dir)


def test_load_scene_aligns_and_masks(synthetic_scene_dir):
    scene = load_scene(synthetic_scene_dir)
    assert scene.scene_id == SCENE_ID
    # IR was 64x64 @ 60 m; it must be resampled onto the 128x128 @ 30 m grid
    assert "ir" in scene.resampled_bands
    assert scene.ir_kelvin.shape == (SIZE, SIZE)
    assert scene.rgb_reflectance.shape == (3, SIZE, SIZE)

    # fill columns and the cloud block are invalid, everything else valid
    assert not scene.valid[:, :FILL_COLS].any()
    assert not scene.valid[CLOUD_BLOCK].any()
    assert scene.valid[60:100, 60:100].all()

    # invalid -> NaN, valid values are in physical units
    assert np.isnan(scene.ir_kelvin[~scene.valid]).all()
    ir_valid = scene.ir_kelvin[scene.valid]
    assert 289 < ir_valid.min() and ir_valid.max() < 321
    rgb_valid = scene.rgb_reflectance[:, scene.valid]
    assert 0.04 < rgb_valid.min() and rgb_valid.max() < 0.26

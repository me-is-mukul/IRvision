import copy

import cv2
import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

from irvision.inference.io import read_image, to_png_bytes, to_single_channel
from irvision.inference.pipeline import IRVisionPipeline
from irvision.models.baseline import GrayColorizer
from irvision.models.colorizer import UNetColorizer
from irvision.models.unet import UNet
from irvision.utils.config import load_config
from tests.conftest import textured_field


@pytest.fixture
def cfg():
    cfg = copy.deepcopy(load_config())
    cfg["optional"]["segmentation"]["enabled"] = False   # semantic stage is tested in test_semantic.py
    cfg["optional"]["super_resolution"]["enabled"] = False  # advanced stages: tests/test_advanced.py
    cfg["optional"]["detection"]["enabled"] = False
    return cfg


@pytest.fixture
def pipeline(cfg):
    return IRVisionPipeline(cfg, colorizer=GrayColorizer(), device="cpu")


def kelvin_image(size=128):
    return (290 + 30 * textured_field(size)).astype(np.float32)


def test_process_image_outputs(pipeline):
    res = pipeline.process_image(kelvin_image())
    assert res.colorized.shape == (3, 128, 128) and res.colorized.dtype == np.float32
    assert res.enhanced.shape == res.input.shape == (128, 128)
    assert 0 <= res.colorized.min() and res.colorized.max() <= 1
    assert res.semantic_map is None and res.detections is None
    for stage in ("validation", "normalization", "enhancement", "colorization", "total"):
        assert stage in res.metrics["time_ms"]
    assert "psnr" not in res.metrics          # no reference -> no fabricated metrics


def test_reference_metrics_and_nodata(pipeline):
    ir = kelvin_image()
    ir[:10] = np.nan                          # NaN = invalid
    nodata = np.zeros(ir.shape, bool)
    nodata[:, :5] = True
    ref = np.random.default_rng(0).random((128, 128, 3)).astype(np.float32)   # HWC is accepted
    res = pipeline.process_image(ir, reference_rgb=ref, nodata_mask=nodata)
    assert not res.valid[:10].any() and not res.valid[:, :5].any()
    assert (res.colorized[:, ~res.valid] == 0).all()
    assert np.isfinite(res.metrics["psnr"]) and -1 <= res.metrics["ssim"] <= 1


def test_reference_shape_mismatch_warns_not_crashes(pipeline):
    res = pipeline.process_image(kelvin_image(), reference_rgb=np.zeros((3, 64, 64), np.float32))
    assert "psnr" not in res.metrics
    assert any("does not match" in w for w in res.warnings)


def test_rgb_input_is_converted_with_warning(pipeline):
    img = (np.random.default_rng(0).random((64, 64, 3)) * 255).astype(np.uint8)
    res = pipeline.process_image(img)
    assert res.colorized.shape == (3, 64, 64)
    assert any("grayscale" in w for w in res.warnings)


@pytest.mark.parametrize("bad", [np.zeros((10, 10)), np.zeros((4, 64, 64)), np.full((64, 64), np.nan)])
def test_invalid_inputs_rejected(pipeline, bad):
    with pytest.raises(ValueError):
        pipeline.process_image(bad)


def test_enabled_optional_stage_is_skipped_safely(cfg):
    cfg["optional"]["super_resolution"]["enabled"] = True
    pipe = IRVisionPipeline(cfg, colorizer=GrayColorizer(), device="cpu")
    pipe._optional_models["super_resolution"] = (None, "model unavailable")   # simulate a failed load
    res = pipe.process_image(kelvin_image())
    assert res.colorized.shape == (3, 128, 128) and res.scale == 1
    assert any("Super resolution skipped" in w for w in res.warnings)


def test_fallback_colorizer_when_no_model(cfg, tmp_path):
    cfg["inference"]["checkpoint"] = str(tmp_path / "missing.pt")
    cfg["paths"]["models_dir"] = tmp_path
    pipe = IRVisionPipeline(cfg, device="cpu")
    assert pipe.colorizer_info["fallback"] and pipe.colorizer_info["name"] == "colormap"
    res = pipe.process_image(kelvin_image())
    assert any("fallback" in w for w in res.warnings)


def test_unet_colorizer_pads_and_tiles():
    torch.manual_seed(0)
    col = UNetColorizer(UNet(base_channels=4, depth=2), tile_size=64, tile_overlap=16)
    small = col.colorize(textured_field(50))                      # padded 50 -> 52
    assert small.shape == (3, 50, 50)
    ir = np.random.default_rng(0).random((150, 97)).astype(np.float32)
    big = col.colorize(ir)                                        # tiled path
    assert big.shape == (3, 150, 97) and np.isfinite(big).all()
    # away from tile borders, tiled output equals a single full pass
    col.tile_size = 1024
    full = col.colorize(ir)
    np.testing.assert_allclose(big[:, 20:40, 20:40], full[:, 20:40, 20:40], atol=1e-4)


def test_read_image_png16_and_geotiff_nodata(tmp_path):
    png = (textured_field(32) * 65535).astype(np.uint16)
    ok, buf = cv2.imencode(".png", png)
    arr, mask = read_image(buf.tobytes(), "x.png")
    assert arr.dtype == np.uint16 and mask is None
    np.testing.assert_array_equal(arr, png)

    data = np.full((16, 16), 500, np.uint16)
    data[0] = 0
    path = tmp_path / "LC09_X_ST_B10.TIF"
    with rasterio.open(path, "w", driver="GTiff", width=16, height=16, count=1, dtype="uint16",
                       crs="EPSG:32643", transform=from_origin(0, 0, 30, 30)) as dst:
        dst.write(data, 1)
    arr, mask = read_image(path)
    assert arr.shape == (16, 16) and mask[0].all() and not mask[1:].any()   # Landsat fill = 0


def test_png_roundtrip_and_single_channel():
    rgb = np.random.default_rng(0).random((3, 8, 8)).astype(np.float32)
    decoded = cv2.imdecode(np.frombuffer(to_png_bytes(rgb), np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (8, 8, 3)
    gray_as_rgb = np.repeat(np.arange(16, dtype=np.uint8).reshape(4, 4, 1), 3, axis=2)
    out, warning = to_single_channel(gray_as_rgb)
    assert out.shape == (4, 4) and warning is None


def test_read_geotiff_with_nan_nodata(tmp_path):
    data = np.full((8, 8), 300.0, np.float32)
    data[0] = np.nan
    path = tmp_path / "ir.tif"
    with rasterio.open(path, "w", driver="GTiff", width=8, height=8, count=1, dtype="float32", nodata=np.nan,
                       crs="EPSG:32643", transform=from_origin(0, 0, 30, 30)) as dst:
        dst.write(data, 1)
    arr, mask = read_image(path)
    assert mask[0].all() and not mask[1:].any()


def test_colorizer_tta_shape_and_range():
    """TTA averages 8 flipped/rotated passes; output keeps the input size and value range."""
    torch.manual_seed(0)
    model = UNet(base_channels=4, depth=2)
    col = UNetColorizer(model, tile_size=64)
    ir = np.random.default_rng(0).random((48, 40)).astype(np.float32)
    plain = col.colorize(ir)
    col.tta = True
    tta = col.colorize(ir)
    assert tta.shape == plain.shape == (3, 48, 40)
    assert np.isfinite(tta).all() and 0 <= tta.min() and tta.max() <= 1

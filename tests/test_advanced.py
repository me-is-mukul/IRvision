"""Advanced features: perceptual loss, super-resolution, detection. Offline (no pre-trained downloads)."""

import copy

import numpy as np
import pytest
import torch

from irvision.detection.detector import AerialDetector
from irvision.detection.metrics import box_iou, compare_detections, match_counts
from irvision.inference.pipeline import IRVisionPipeline
from irvision.models.baseline import GrayColorizer
from irvision.models.super_resolution import SuperResolver
from irvision.training.losses import ColorizationLoss, VGGPerceptualLoss
from irvision.utils.config import load_config
from irvision.utils.tiling import tiled_apply
from tests.conftest import textured_field


# -- perceptual loss -----------------------------------------------------------------
def test_perceptual_loss_zero_when_identical_and_masked():
    loss = VGGPerceptualLoss(pretrained=False)
    y = torch.rand(2, 3, 32, 32)
    valid = torch.ones(2, 1, 32, 32)
    assert loss(y, y, valid).item() == 0.0
    pred = y.clone()
    pred[..., :8, :] = 1 - pred[..., :8, :]
    valid[..., :8, :] = 0                       # the changed rows are invalid -> no loss
    assert loss(pred, y, valid).item() == pytest.approx(0.0, abs=1e-6)
    assert loss(pred, y, torch.ones_like(valid)).item() > 0


def test_perceptual_loss_vgg_frozen_but_pred_gets_gradient():
    lf = ColorizationLoss(l1=1.0, ssim=0.0, perceptual=0.1, pretrained_vgg=False)
    lf.train()
    assert not lf.perceptual.blocks.training                 # VGG stays in eval mode
    pred = torch.rand(1, 3, 32, 32, requires_grad=True)
    total, parts = lf(pred, torch.rand(1, 3, 32, 32), torch.ones(1, 1, 32, 32))
    total.backward()
    assert "perceptual" in parts and pred.grad is not None
    assert all(p.grad is None for p in lf.perceptual.parameters())


# -- super-resolution ------------------------------------------------------------------
class Upsample2x(torch.nn.Module):
    """Stand-in for EDSR: nearest-neighbour x2 on RGB input."""

    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")


def test_tiled_apply_with_scale_matches_single_pass():
    img = np.random.default_rng(0).random((1, 150, 97)).astype(np.float32)
    up = lambda t: np.repeat(np.repeat(t, 2, axis=1), 2, axis=2)  # noqa: E731
    np.testing.assert_allclose(tiled_apply(up, img, 1, tile_size=64, overlap=16, scale=2), up(img), rtol=1e-5)


def test_super_resolver_shape_and_range():
    sr = SuperResolver(Upsample2x(), scale=2, tile_size=32)
    img = textured_field(50)
    out = sr.upscale(img)
    assert out.shape == (100, 100) and out.dtype == np.float32
    np.testing.assert_allclose(out[::2, ::2], img, atol=1e-6)


# -- detection -----------------------------------------------------------------------------
def test_box_iou_and_matching():
    a = {"class_name": "ship", "confidence": 0.9, "box": [0, 0, 10, 10]}
    b = {"class_name": "ship", "confidence": 0.8, "box": [5, 0, 15, 10]}
    assert box_iou(a["box"], a["box"]) == 1.0
    assert box_iou(a["box"], b["box"]) == pytest.approx(1 / 3)
    assert match_counts([a], [a])["tp"] == 1
    assert match_counts([b], [a])["tp"] == 0                          # IoU 0.33 < 0.5
    assert match_counts([{**a, "class_name": "plane"}], [a])["tp"] == 0   # class must match
    r = compare_detections([a, b], [a])
    assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 0) and r["precision"] == 0.5 and r["recall"] == 1.0
    empty = compare_detections([], [])
    assert empty["precision"] is None and empty["recall"] is None


class FakeDetector(AerialDetector):
    """Reports one 'ship' at the centre of every tile, so tile offsets can be checked."""

    def __init__(self, tile_size):
        self.tile_size, self.class_names = tile_size, {0: "ship"}

    def _detect_tile(self, rgb):
        h, w = rgb.shape[1:]
        return [{"class_name": "ship", "confidence": 0.5, "box": [w / 2 - 1, h / 2 - 1, w / 2 + 1, h / 2 + 1],
                 "polygon": [[0, 0]] * 4}]


def test_detector_tiles_and_offsets_boxes():
    boxes = FakeDetector(tile_size=64).detect(np.zeros((3, 128, 128), np.float32))
    centres = sorted(((b["box"][0] + b["box"][2]) / 2, (b["box"][1] + b["box"][3]) / 2) for b in boxes)
    assert centres == [(32, 32), (32, 96), (96, 32), (96, 96)]


# -- pipeline integration ------------------------------------------------------------------
def test_pipeline_with_super_resolution_and_detection():
    cfg = copy.deepcopy(load_config())
    cfg["optional"]["segmentation"]["enabled"] = False
    pipe = IRVisionPipeline(cfg, colorizer=GrayColorizer(), device="cpu")
    pipe._optional_models["super_resolution"] = (SuperResolver(Upsample2x(), scale=2), "")
    pipe._optional_models["detection"] = (FakeDetector(tile_size=64), "")
    ir = (290 + 30 * textured_field(64)).astype(np.float32)
    ref = np.random.default_rng(0).random((3, 64, 64)).astype(np.float32)

    res = pipe.process_image(ir, reference_rgb=ref, super_resolution=True, detection=True)
    assert res.scale == 2 and res.colorized.shape == (3, 128, 128) and res.super_resolved.shape == (128, 128)
    assert np.isfinite(res.metrics["psnr"])                  # scored at the input resolution
    assert len(res.detections) == 4 and len(res.reference_detections) == 4
    assert res.metrics["detection"]["tp"] == 4                # same fake objects -> all matched
    assert "super_resolution" in res.metrics["time_ms"] and "detection" in res.metrics["time_ms"]

    plain = pipe.process_image(ir, super_resolution=False, detection=False)
    assert plain.scale == 1 and plain.detections is None and plain.colorized.shape == (3, 64, 64)


def test_patch_discriminator_and_losses():
    from irvision.training.gan import PatchDiscriminator, d_loss, g_adv_loss

    disc = PatchDiscriminator(in_channels=4)
    scores = disc(torch.rand(2, 1, 64, 64), torch.rand(2, 3, 64, 64))
    assert scores.ndim == 4 and scores.shape[:2] == (2, 1)
    ones, zeros = torch.ones(2, 1, 4, 4), torch.zeros(2, 1, 4, 4)
    assert d_loss(ones, zeros).item() == 0.0 and g_adv_loss(ones).item() == 0.0

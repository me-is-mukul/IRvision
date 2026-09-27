import copy

import numpy as np
import pytest
import torch

from irvision.inference.pipeline import IRVisionPipeline
from irvision.models.baseline import GrayColorizer
from irvision.semantic.landcover import CLASSES, IGNORE, colorize_labels, remap_worldcover
from irvision.semantic.metrics import compare_maps, confusion_matrix, summarize
from irvision.semantic.segmenter import LandCoverSegmenter, build_segmenter
from irvision.utils.config import load_config
from irvision.utils.tiling import tiled_apply
from tests.conftest import textured_field


def test_remap_worldcover():
    codes = np.array([10, 40, 50, 60, 80, 0, 95, 123], np.uint8)
    assert remap_worldcover(codes).tolist() == [0, 1, 2, 3, 4, IGNORE, 0, IGNORE]
    img = colorize_labels(np.array([[0, IGNORE]], np.uint8))
    assert img.shape == (1, 2, 3) and (img[0, 1] == 0).all()


def test_metrics_perfect_and_partial():
    a = np.array([[0, 0, 1, 1]], np.uint8)
    perfect = compare_maps(a, a)
    assert perfect["miou"] == 1.0 and perfect["pixel_agreement"] == 1.0
    b = np.array([[0, 1, 1, 1]], np.uint8)            # one class-0 pixel predicted as 1
    s = compare_maps(b, a)
    assert s["iou"]["tree cover"] == 0.5             # tp 1, fn 1
    assert s["iou"]["low vegetation / crops"] == pytest.approx(2 / 3, abs=1e-4)   # reported to 4 decimals
    assert s["pixel_agreement"] == 0.75
    assert s["iou"]["water"] is None                  # absent class: not averaged


def test_confusion_matrix_skips_ignore_and_accumulates():
    a = np.array([[0, IGNORE, 2]], np.uint8)
    cm = confusion_matrix(a, a) + confusion_matrix(a, a)
    assert cm.sum() == 4 and summarize(cm)["miou"] == 1.0


def test_tiled_apply_matches_single_pass_for_pointwise_fn():
    img = np.random.default_rng(0).random((3, 300, 200)).astype(np.float32)
    out = tiled_apply(lambda t: t * 2, img, 3, tile_size=128, overlap=32)
    np.testing.assert_allclose(out, img * 2, rtol=1e-5)


def test_segmenter_predict_shape_and_ignore(tmp_path):
    torch.manual_seed(0)
    seg = LandCoverSegmenter(build_segmenter(pretrained_backbone=False))
    rgb = np.stack([textured_field(64, seed=s) for s in range(3)])
    valid = np.ones((64, 64), bool)
    valid[:5] = False
    labels = seg.predict(rgb, valid)
    assert labels.shape == (64, 64) and labels.dtype == np.uint8
    assert (labels[:5] == IGNORE).all() and (labels[5:] < len(CLASSES)).all()

    torch.save({"state_dict": seg.model.state_dict(), "classes": CLASSES, "epoch": 1, "val": {}}, tmp_path / "s.pt")
    loaded = LandCoverSegmenter.from_checkpoint(tmp_path / "s.pt")
    np.testing.assert_array_equal(loaded.predict(rgb, valid), labels)


class ConstantSegmenter:
    """Predicts class 1 everywhere: deterministic stand-in for the real model."""

    def predict(self, rgb, valid=None):
        out = np.ones(rgb.shape[1:], np.uint8)
        if valid is not None:
            out[~valid] = IGNORE
        return out


def test_pipeline_semantic_validation():
    cfg = copy.deepcopy(load_config())
    cfg["optional"]["segmentation"]["enabled"] = True
    cfg["optional"]["detection"]["enabled"] = False
    pipe = IRVisionPipeline(cfg, colorizer=GrayColorizer(), device="cpu", segmenter=ConstantSegmenter())
    ir = (290 + 30 * textured_field(64)).astype(np.float32)
    ref = np.random.default_rng(0).random((3, 64, 64)).astype(np.float32)

    res = pipe.process_image(ir, reference_rgb=ref)
    assert res.semantic_map.shape == (64, 64) and res.reference_semantic_map.shape == (64, 64)
    assert res.metrics["semantic"]["miou"] == 1.0              # identical maps
    assert "segmentation" in res.metrics["time_ms"]

    no_ref = pipe.process_image(ir)
    assert no_ref.semantic_map is not None and "semantic" not in no_ref.metrics


def test_pipeline_segmentation_missing_checkpoint_warns(tmp_path):
    cfg = copy.deepcopy(load_config())
    cfg["optional"]["segmentation"]["enabled"] = True
    cfg["semantic"]["checkpoint"] = str(tmp_path / "missing.pt")
    res = IRVisionPipeline(cfg, colorizer=GrayColorizer(), device="cpu").process_image(
        (290 + 30 * textured_field(64)).astype(np.float32))
    assert res.semantic_map is None
    assert any("Semantic segmentation skipped" in w for w in res.warnings)


def test_landcover_patch_dataset_shapes(tmp_path):
    """Regression: labels are stored (1, S, S) and must come out (S, S) for cross-entropy."""
    from irvision.preprocessing.patches import Patch, save_patch
    from irvision.semantic.train import LandCoverPatches

    ir = textured_field(32)
    lc = np.ones((32, 32), np.uint8)
    valid = np.ones((32, 32), bool)
    valid[0] = False
    save_patch(tmp_path / "S_r0_c0.npz", Patch(0, 0, "train", 0.0, ir[None], ir[None], np.stack([ir] * 3), valid,
                                               extra={"landcover": lc[None]}))
    rgb, labels = LandCoverPatches(tmp_path)[0]
    assert rgb.shape == (3, 32, 32) and labels.shape == (32, 32) and labels.dtype == torch.int64
    assert (labels[0] == IGNORE).all() and (labels[1:] == 1).all()

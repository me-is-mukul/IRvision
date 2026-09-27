import numpy as np
import pytest
import torch

from irvision.models.unet import UNet, build_unet, load_checkpoint
from irvision.training.augment import RandomFlipRotate
from irvision.training.losses import ColorizationLoss, masked_l1, masked_ssim


def test_unet_forward_shape():
    """PLAN.md §15 minimum check."""
    model = UNet(in_channels=1, out_channels=3, base_channels=8)
    dummy_input = torch.randn(1, 1, 256, 256)
    output = model(dummy_input)
    assert output.shape == (1, 3, 256, 256)
    assert 0 <= output.min() and output.max() <= 1   # sigmoid output


def test_unet_multichannel_input():
    model = build_unet({"out_channels": 3, "base_channels": 8, "depth": 3}, in_channels=2)
    assert model(torch.rand(2, 2, 64, 64)).shape == (2, 3, 64, 64)


def test_unet_rejects_bad_size():
    with pytest.raises(ValueError, match="divisible"):
        UNet(base_channels=8)(torch.rand(1, 1, 100, 100))


def test_checkpoint_roundtrip(tmp_path):
    cfg = {"out_channels": 3, "base_channels": 8, "depth": 2}
    model = build_unet(cfg, in_channels=1).eval()
    torch.save({"state_dict": model.state_dict(), "model_cfg": cfg, "inputs": ["ir_clahe"], "val_metrics": {}},
               tmp_path / "m.pt")
    loaded, meta = load_checkpoint(tmp_path / "m.pt")
    x = torch.rand(1, 1, 32, 32)
    torch.testing.assert_close(loaded(x), model(x))
    assert meta["inputs"] == ["ir_clahe"]


def test_masked_l1_ignores_invalid():
    pred, target = torch.zeros(1, 3, 8, 8), torch.zeros(1, 3, 8, 8)
    pred[..., :4, :] = 1.0
    valid = torch.ones(1, 1, 8, 8)
    valid[..., :4, :] = 0
    assert masked_l1(pred, target, valid) == 0
    assert masked_l1(pred, target, torch.ones_like(valid)) == pytest.approx(0.5)


def test_masked_ssim_identical_is_one():
    x = torch.rand(2, 3, 32, 32)
    assert masked_ssim(x, x, torch.ones(2, 1, 32, 32)).item() == pytest.approx(1.0, abs=1e-5)
    assert masked_ssim(x, torch.rand_like(x), torch.ones(2, 1, 32, 32)).item() < 0.5


def test_loss_combination():
    x, y, v = torch.rand(1, 3, 16, 16), torch.rand(1, 3, 16, 16), torch.ones(1, 1, 16, 16)
    total, parts = ColorizationLoss(l1=1.0, ssim=0.5)(x, y, v)
    assert set(parts) == {"l1", "ssim"}                        # perceptual: tests/test_advanced.py
    assert total.item() == pytest.approx(parts["l1"] + 0.5 * (1 - parts["ssim"]), rel=1e-5)


def test_augment_keeps_pairs_aligned():
    x = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    aug = RandomFlipRotate(seed=0)
    for _ in range(10):
        ax, ay, av = aug(x, np.repeat(x, 3, 0), x.copy())
        np.testing.assert_array_equal(ax, ay[:1])      # same geometric transform
        np.testing.assert_array_equal(ax, av)

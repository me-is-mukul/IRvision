"""End-to-end check of the training loop on tiny synthetic data (CPU, ~seconds)."""

import copy

import numpy as np

from irvision.models.colorizer import UNetColorizer
from irvision.preprocessing.patches import Patch, save_patch
from irvision.training.trainer import train
from irvision.utils.config import load_config
from tests.conftest import textured_field


def _patches(folder, n, size=32):
    folder.mkdir(parents=True)
    for i in range(n):
        ir = textured_field(size, seed=i)
        rgb = np.stack([ir, 1 - ir, np.full_like(ir, 0.3)])
        save_patch(folder / f"S_r00000_c{i:05d}.npz",
                   Patch(0, i, "train", 0.0, ir[None], ir[None], rgb, np.ones_like(ir, bool)))


def test_train_one_epoch_writes_artifacts(tmp_path):
    cfg = copy.deepcopy(load_config())
    cfg["paths"].update(train_dir=tmp_path / "train", val_dir=tmp_path / "val", models_dir=tmp_path / "models")
    _patches(cfg["paths"]["train_dir"], 4)
    _patches(cfg["paths"]["val_dir"], 2)
    cfg["model"].update(base_channels=4, depth=2)
    cfg["training"].update(epochs=2, batch_size=2, loss_weights={"l1": 1.0, "ssim": 0.5, "perceptual": 0.0})

    result = train(cfg, "tiny", device="cpu")

    run = tmp_path / "models" / "tiny"
    for name in ("best.pt", "last.pt", "history.csv", "curves.png", "summary.json"):
        assert (run / name).exists(), name
    assert len(result.history) == 2 and result.best_epoch in (1, 2)
    col = UNetColorizer.from_checkpoint(run / "best.pt")                # checkpoint is usable
    assert col.colorize(textured_field(40)).shape == (3, 40, 40)

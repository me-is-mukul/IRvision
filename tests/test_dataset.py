import numpy as np
import pytest
import torch

from irvision.preprocessing.patches import Patch, save_patch
from irvision.training.dataset import PatchDataset


def _write_patches(folder, n=3, size=64):
    rng = np.random.default_rng(0)
    for i in range(n):
        patch = Patch(
            row=0, col=i * size, split="train", invalid_fraction=0.0,
            ir=rng.random((1, size, size), dtype=np.float32),
            ir_clahe=rng.random((1, size, size), dtype=np.float32),
            rgb=rng.random((3, size, size), dtype=np.float32),
            valid=np.ones((size, size), bool),
        )
        save_patch(folder / f"scene_r00000_c{i:05d}.npz", patch)


def test_dataset_returns_tensors(tmp_path):
    _write_patches(tmp_path)
    ds = PatchDataset(tmp_path)
    assert len(ds) == 3
    item = ds[0]
    assert item["input"].shape == (1, 64, 64) and item["input"].dtype == torch.float32
    assert item["target"].shape == (3, 64, 64)
    assert item["valid"].shape == (1, 64, 64)
    assert 0 <= item["target"].min() and item["target"].max() <= 1


def test_dataset_works_with_dataloader(tmp_path):
    _write_patches(tmp_path)
    batch = next(iter(torch.utils.data.DataLoader(PatchDataset(tmp_path), batch_size=2)))
    assert batch["input"].shape == (2, 1, 64, 64)


def test_dataset_stacks_inputs_and_caches(tmp_path):
    _write_patches(tmp_path)
    ds = PatchDataset(tmp_path, inputs=["ir_clahe", "ir"], cache=True)
    assert ds[1]["input"].shape == (2, 64, 64)
    with pytest.raises(KeyError, match="ir_abs"):
        PatchDataset(tmp_path, inputs=["ir_abs"])[0]


def test_dataset_empty_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        PatchDataset(tmp_path)


def test_dataset_land_cover_labels_follow_augmentation(tmp_path):
    from irvision.training.augment import RandomFlipRotate

    size = 16
    ir = np.arange(size * size, dtype=np.float32).reshape(1, size, size) / (size * size)
    valid = np.ones((size, size), bool)
    valid[0, 0] = False
    lc = (np.arange(size * size).reshape(size, size) % 5).astype(np.uint8)
    save_patch(tmp_path / "s_r0_c0.npz", Patch(0, 0, "train", 0.0, ir, ir, np.repeat(ir, 3, 0), valid,
                                                extra={"landcover": lc[None]}))
    item = PatchDataset(tmp_path, labels=True)[0]
    assert item["labels"].shape == (size, size) and item["labels"].dtype == torch.int64
    assert item["labels"][0, 0] == 255                        # invalid pixel ignored
    aug = PatchDataset(tmp_path, labels=True, transform=RandomFlipRotate(seed=3))[0]
    # the label map moves exactly like the input image
    lab_of = {float(v): int(l) for v, l in zip(item["input"][0].flatten(), item["labels"].flatten())}
    for v, l in zip(aug["input"][0].flatten(), aug["labels"].flatten()):
        assert lab_of[float(v)] == int(l)

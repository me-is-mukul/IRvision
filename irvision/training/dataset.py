"""PyTorch Dataset over the patch files written by ``scripts/prepare_dataset.py``."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PatchDataset(Dataset):
    """Returns dicts ``{"input": (C,S,S), "target": (3,S,S), "valid": (1,S,S), "name": str}``.

    ``inputs`` lists the patch keys stacked as input channels, e.g.
    ``["ir_clahe"]`` (default, matches the inference pipeline) or
    ``["ir_clahe", "ir_abs"]``. ``transform`` receives and returns
    ``(input, target, valid)`` numpy arrays (augmentation).
    ``cache=True`` keeps all patches in RAM (float16) so epochs don't re-read
    and decompress files; ~0.5 MB per 256px patch.
    """

    def __init__(
        self,
        split_dir: str | Path,
        inputs: Sequence[str] = ("ir_clahe",),
        transform: Callable | None = None,
        cache: bool = False,
    ):
        self.files = sorted(Path(split_dir).glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz patches in {split_dir}. Run scripts/prepare_dataset.py first.")
        self.inputs = list(inputs)
        self.transform = transform
        self._cache = [self._read(i) for i in range(len(self.files))] if cache else None

    def _read(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with np.load(self.files[idx]) as data:
            missing = [k for k in self.inputs if k not in data]
            if missing:
                raise KeyError(f"{self.files[idx].name} has no {missing}; re-run scripts/prepare_dataset.py")
            x = np.concatenate([data[k] for k in self.inputs], axis=0)
            return x, data["rgb"], data["valid"][None]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        x, y, valid = self._cache[idx] if self._cache is not None else self._read(idx)
        x, y, valid = x.astype(np.float32), y.astype(np.float32), valid.astype(np.float32)
        if self.transform is not None:
            x, y, valid = self.transform(x, y, valid)
        return {
            "input": torch.from_numpy(np.ascontiguousarray(x)),
            "target": torch.from_numpy(np.ascontiguousarray(y)),
            "valid": torch.from_numpy(np.ascontiguousarray(valid)),
            "name": self.files[idx].stem,
        }

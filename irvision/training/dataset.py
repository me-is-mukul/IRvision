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
    ``labels=True`` adds ``"labels"``: (S,S) int64 land-cover classes, 255 on invalid pixels
    (for the colorizer's auxiliary land-cover head).
    """

    def __init__(
        self,
        split_dir: str | Path,
        inputs: Sequence[str] = ("ir_clahe",),
        transform: Callable | None = None,
        cache: bool = False,
        labels: bool = False,
    ):
        self.files = sorted(Path(split_dir).glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz patches in {split_dir}. Run scripts/prepare_dataset.py first.")
        self.inputs = list(inputs)
        self.transform = transform
        self.labels = labels
        self._cache = [self._read(i) for i in range(len(self.files))] if cache else None

    def _read(self, idx: int) -> tuple[np.ndarray, ...]:
        with np.load(self.files[idx]) as data:
            needed = self.inputs + (["landcover"] if self.labels else [])
            missing = [k for k in needed if k not in data]
            if missing:
                raise KeyError(f"{self.files[idx].name} has no {missing}; re-run scripts/prepare_dataset.py")
            x = np.concatenate([data[k] for k in self.inputs], axis=0)
            if not self.labels:
                return x, data["rgb"], data["valid"][None]
            labels = np.where(data["valid"], data["landcover"][0], 255).astype(np.uint8)[None]
            return x, data["rgb"], data["valid"][None], labels

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        arrays = self._cache[idx] if self._cache is not None else self._read(idx)
        x, y, valid = (a.astype(np.float32) for a in arrays[:3])
        extra = list(arrays[3:])
        if self.transform is not None:
            x, y, valid, *extra = self.transform(x, y, valid, *extra)
        item = {
            "input": torch.from_numpy(np.ascontiguousarray(x)),
            "target": torch.from_numpy(np.ascontiguousarray(y)),
            "valid": torch.from_numpy(np.ascontiguousarray(valid)),
            "name": self.files[idx].stem,
        }
        if self.labels:
            item["labels"] = torch.from_numpy(np.ascontiguousarray(extra[0][0])).long()
        return item

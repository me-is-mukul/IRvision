"""U-Net wrapped in the common colorizer interface (see models/baseline.py).

    UNetColorizer.from_checkpoint("outputs/models/unet_l1/best.pt").colorize(ir)

Handles any image size: sides are padded (reflect) to a multiple of 16, and
images larger than ``tile_size`` are processed in overlapping tiles blended
with a feathered weight so no seams appear.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from irvision.models.unet import UNet, load_checkpoint
from irvision.utils.tiling import tiled_apply


class UNetColorizer:
    name = "unet"

    def __init__(self, model: UNet, device: str = "cpu", tile_size: int = 512, tile_overlap: int = 64,
                 meta: dict | None = None, tta: bool = False):
        self.model = model.to(device).eval()
        self.device = device
        self.tile_size, self.tile_overlap = tile_size, tile_overlap
        self.tta = tta   # test-time augmentation: average over the 8 flips/rotations
        self.meta = meta or {}
        self.multiple = 2**model.depth
        inputs = self.meta.get("inputs", ["ir_clahe"])
        if inputs not in (["ir_clahe"], ["ir_clahe", "ir_abs"]):
            raise ValueError(f"Unsupported checkpoint inputs {inputs}")
        self.needs_abs = inputs == ["ir_clahe", "ir_abs"]   # 2nd channel: absolute temperature (Kelvin, range-normalized)

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "cpu", **kwargs) -> "UNetColorizer":
        model, meta = load_checkpoint(path, device)
        meta["checkpoint"] = str(path)
        return cls(model, device, meta=meta, **kwargs)

    @torch.no_grad()
    def _predict(self, ir: np.ndarray) -> np.ndarray:
        """One forward pass on a (C, H, W) image, padding to a multiple of 16."""
        h, w = ir.shape[1:]
        x = torch.from_numpy(np.ascontiguousarray(ir, dtype=np.float32))[None].to(self.device)
        ph, pw = (-h) % self.multiple, (-w) % self.multiple
        if ph or pw:
            mode = "reflect" if ph < h and pw < w else "replicate"
            x = F.pad(x, (0, pw, 0, ph), mode=mode)
        return self.model(x)[0, :, :h, :w].float().cpu().numpy()

    def _colorize_once(self, ir: np.ndarray) -> np.ndarray:
        return tiled_apply(self._predict, ir, 3, self.tile_size, self.tile_overlap)

    def colorize(self, ir: np.ndarray, ir_abs: np.ndarray | None = None) -> np.ndarray:
        """(H, W) float in [0, 1] -> (3, H, W) float32 RGB in [0, 1].

        ``ir_abs``: (H, W) absolute temperature channel, required by 2-input checkpoints.
        """
        ir = np.clip(np.nan_to_num(ir), 0, 1).astype(np.float32)[None]
        if self.needs_abs:
            if ir_abs is None:
                raise ValueError("This checkpoint needs the absolute-temperature channel (ir_abs)")
            ir = np.concatenate([ir, np.clip(np.nan_to_num(ir_abs), 0, 1).astype(np.float32)[None]])
        if not self.tta:
            return self._colorize_once(ir)
        out = np.zeros((3, *ir.shape[1:]), np.float32)
        for k in range(4):
            for flip in (False, True):
                t = np.rot90(ir, k, axes=(1, 2))
                t = t[:, :, ::-1] if flip else t
                pred = self._colorize_once(np.ascontiguousarray(t))
                pred = pred[:, :, ::-1] if flip else pred          # undo in reverse order
                out += np.rot90(pred, -k, axes=(1, 2))
        return out / 8.0

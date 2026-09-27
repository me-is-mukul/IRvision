"""Pre-trained super-resolution (PLAN.md §4A): EDSR x2 from the `super-image` package.

    sr = SuperResolver.load(device="cuda")
    big = sr.upscale(ir)          # (H, W) in [0, 1] -> (2H, 2W) float32 in [0, 1]

EDSR-base (1.4 M params) was trained on natural RGB photos (DIV2K); it is not
trained by this project. The single IR channel is replicated to RGB and the
output channels are averaged back to one.
"""

from __future__ import annotations

import numpy as np
import torch

from irvision.utils.tiling import tiled_apply

DEFAULT_MODEL = "eugenesiow/edsr-base"


class SuperResolver:
    def __init__(self, model: torch.nn.Module, scale: int, device: str = "cpu", tile_size: int = 256, name: str = "edsr"):
        self.model = model.to(device).eval()
        self.scale, self.device, self.tile_size = scale, device, tile_size
        self.name = f"{name}_x{scale}"

    @classmethod
    def load(cls, model_id: str = DEFAULT_MODEL, scale: int = 2, device: str = "cpu") -> "SuperResolver":
        """Load pre-trained EDSR weights (downloaded once into the Hugging Face cache)."""
        from super_image import EdsrModel

        return cls(EdsrModel.from_pretrained(model_id, scale=scale), scale, device, name=model_id.split("/")[-1])

    @torch.no_grad()
    def _run(self, tile: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(np.repeat(tile, 3, axis=0), dtype=np.float32))[None].to(self.device)
        return self.model(x)[0].mean(dim=0, keepdim=True).clamp(0, 1).float().cpu().numpy()

    def upscale(self, image: np.ndarray) -> np.ndarray:
        """(H, W) float in [0, 1] -> (H*scale, W*scale) float32 in [0, 1]."""
        img = np.clip(np.nan_to_num(image), 0, 1).astype(np.float32)[None]
        return tiled_apply(self._run, img, 1, self.tile_size, overlap=16, scale=self.scale)[0]

"""Land-cover segmenter: DeepLabV3 with an ImageNet-pretrained MobileNetV3 backbone.

    seg = LandCoverSegmenter.from_checkpoint("outputs/models/segmenter/best.pt", "cuda")
    labels = seg.predict(rgb)      # rgb (3, H, W) in [0, 1] -> (H, W) uint8 class indices

Inputs use the project's RGB normalization (reflectance 0-0.3 -> [0, 1]); the
ImageNet mean/std normalization happens inside ``predict``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

from irvision.semantic.landcover import CLASSES, IGNORE
from irvision.utils.tiling import tiled_apply

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def build_segmenter(num_classes: int = len(CLASSES), pretrained_backbone: bool = True) -> torch.nn.Module:
    """DeepLabV3-MobileNetV3; backbone weights from ImageNet (downloaded once, ~21 MB)."""
    return deeplabv3_mobilenet_v3_large(
        weights=None,
        weights_backbone="IMAGENET1K_V1" if pretrained_backbone else None,
        num_classes=num_classes,
        aux_loss=False,
    )


def normalize_input(rgb: torch.Tensor) -> torch.Tensor:
    """(B, 3, H, W) in [0, 1] -> ImageNet-normalized."""
    return (rgb - IMAGENET_MEAN.to(rgb.device)) / IMAGENET_STD.to(rgb.device)


class LandCoverSegmenter:
    def __init__(self, model: torch.nn.Module, device: str = "cpu", tile_size: int = 1024, meta: dict | None = None):
        self.model = model.to(device).eval()
        self.device = device
        self.tile_size = tile_size
        self.meta = meta or {}

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "cpu", **kwargs) -> "LandCoverSegmenter":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = build_segmenter(len(ckpt["classes"]), pretrained_backbone=False)
        model.load_state_dict(ckpt["state_dict"])
        meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
        meta["checkpoint"] = str(path)
        return cls(model, device, meta=meta, **kwargs)

    @torch.no_grad()
    def _logits(self, rgb: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(rgb, dtype=np.float32))[None].to(self.device)
        return self.model(normalize_input(x))["out"][0].float().cpu().numpy()

    def predict(self, rgb: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
        """(3, H, W) RGB in [0, 1] -> (H, W) uint8 labels; IGNORE where ``valid`` is False."""
        rgb = np.clip(np.nan_to_num(rgb), 0, 1).astype(np.float32)
        logits = tiled_apply(self._logits, rgb, len(CLASSES), self.tile_size, overlap=128)
        labels = logits.argmax(axis=0).astype(np.uint8)
        if valid is not None:
            labels[~valid] = IGNORE
        return labels

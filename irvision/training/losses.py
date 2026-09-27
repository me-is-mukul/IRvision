"""Masked losses: invalid pixels (cloud, shadow, no-data) never contribute.

``valid`` is a (B, 1, H, W) float tensor with 1 = use, 0 = ignore.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

C1, C2 = 0.01**2, 0.03**2     # standard SSIM constants for data range 1


def masked_l1(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    err = (pred - target).abs() * valid
    return err.sum() / (valid.sum() * pred.shape[1]).clamp_min(1.0)


def ssim_map(x: torch.Tensor, y: torch.Tensor, window: int = 7) -> torch.Tensor:
    """Per-pixel SSIM with a uniform ``window`` (no padding: output is smaller by window-1)."""
    mu_x, mu_y = F.avg_pool2d(x, window, 1), F.avg_pool2d(y, window, 1)
    var_x = F.avg_pool2d(x * x, window, 1) - mu_x**2
    var_y = F.avg_pool2d(y * y, window, 1) - mu_y**2
    cov = F.avg_pool2d(x * y, window, 1) - mu_x * mu_y
    return ((2 * mu_x * mu_y + C1) * (2 * cov + C2)) / ((mu_x**2 + mu_y**2 + C1) * (var_x + var_y + C2))


def masked_ssim(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, window: int = 7) -> torch.Tensor:
    """Mean SSIM over windows whose pixels are *all* valid."""
    smap = ssim_map(pred.float(), target.float(), window).mean(dim=1, keepdim=True)
    window_valid = (F.avg_pool2d(valid, window, 1) > 0.999).float()
    return (smap * window_valid).sum() / window_valid.sum().clamp_min(1.0)


class VGGPerceptualLoss(nn.Module):
    """L1 distance between frozen VGG19 feature maps (relu1_2, relu2_2, relu3_4) — PLAN §4D.

    Invalid pixels of ``pred`` are replaced by the target before feature extraction,
    so cloud / no-data regions contribute nothing.
    """

    SLICES = ((0, 4), (4, 9), (9, 18))   # vgg19.features indices ending at relu1_2, relu2_2, relu3_4
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision.models import VGG19_Weights, vgg19

        features = vgg19(weights=VGG19_Weights.IMAGENET1K_V1 if pretrained else None).features[:18].eval()
        for p in features.parameters():
            p.requires_grad_(False)
        self.blocks = nn.ModuleList(features[a:b] for a, b in self.SLICES)
        self.register_buffer("mean", torch.tensor(self.MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(self.STD).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        return super().train(False)          # VGG stays in eval mode

    def forward(self, pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        pred = pred * valid + target * (1 - valid)
        x, y = (pred - self.mean) / self.std, (target - self.mean) / self.std
        loss = pred.new_zeros(())
        for block in self.blocks:
            x, y = block(x), block(y)
            loss = loss + F.l1_loss(x, y)
        return loss / len(self.blocks)


class ColorizationLoss(nn.Module):
    """``w_l1 * L1 + w_ssim * (1 - SSIM) + w_perceptual * VGG19 + w_aux * CE(land cover)``.

    Colour terms are masked by ``valid``; the land-cover term ignores label 255.
    Move to the training device with ``.to(device)`` (the VGG weights live inside).
    """

    def __init__(self, l1: float = 1.0, ssim: float = 0.0, perceptual: float = 0.0, aux: float = 0.0,
                 pretrained_vgg: bool = True, class_weights: torch.Tensor | None = None):
        super().__init__()
        self.w_l1, self.w_ssim, self.w_perceptual, self.w_aux = l1, ssim, perceptual, aux
        self.perceptual = VGGPerceptualLoss(pretrained_vgg) if perceptual else None
        self.register_buffer("class_weights", class_weights if class_weights is not None else torch.empty(0))

    def forward(self, pred, target, valid, logits=None, labels=None) -> tuple[torch.Tensor, dict[str, float]]:
        parts = {"l1": masked_l1(pred, target, valid)}
        total = self.w_l1 * parts["l1"]
        if self.w_ssim:
            parts["ssim"] = masked_ssim(pred, target, valid)
            total = total + self.w_ssim * (1 - parts["ssim"])
        if self.perceptual is not None:
            parts["perceptual"] = self.perceptual(pred, target, valid)
            total = total + self.w_perceptual * parts["perceptual"]
        if self.w_aux and logits is not None and labels is not None:
            weight = self.class_weights if self.class_weights.numel() else None
            parts["aux"] = F.cross_entropy(logits.float(), labels, weight=weight, ignore_index=255)
            total = total + self.w_aux * parts["aux"]
        return total, {k: float(v.detach()) for k, v in parts.items()}

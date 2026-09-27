"""PatchGAN discriminator for adversarial fine-tuning (Pix2Pix-style).

L1/SSIM training rewards safe, smooth averages: outputs are much less textured than
real imagery (and smooth regions get mistaken for water). A PatchGAN judges whether
each ~70x70 px patch of (IR, RGB) looks real, pushing the colorizer toward realistic
local texture. Least-squares GAN loss (stable, no sigmoid).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils import spectral_norm


class PatchDiscriminator(nn.Module):
    """(IR, RGB) pair -> grid of real/fake scores, one per overlapping ~70 px patch."""

    def __init__(self, in_channels: int = 4, base: int = 64):
        super().__init__()

        def block(i: int, o: int, stride: int, norm: bool = True) -> list[nn.Module]:
            layers: list[nn.Module] = [spectral_norm(nn.Conv2d(i, o, 4, stride, 1))]
            if norm:
                layers.append(nn.InstanceNorm2d(o))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            *block(in_channels, base, 2, norm=False),
            *block(base, base * 2, 2),
            *block(base * 2, base * 4, 2),
            *block(base * 4, base * 8, 1),
            spectral_norm(nn.Conv2d(base * 8, 1, 4, 1, 1)),
        )

    def forward(self, ir: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([ir, rgb], dim=1))


def d_loss(real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
    """LSGAN discriminator loss: real -> 1, fake -> 0."""
    return 0.5 * (((real_scores - 1) ** 2).mean() + (fake_scores**2).mean())


def g_adv_loss(fake_scores: torch.Tensor) -> torch.Tensor:
    """LSGAN generator loss: make fakes score as real (1)."""
    return ((fake_scores - 1) ** 2).mean()

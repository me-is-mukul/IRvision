"""U-Net for IR -> RGB colorization (PLAN.md §8).

Input  (B, in_channels, H, W), values in [0, 1]
Output (B, out_channels, H, W), values in [0, 1] (sigmoid)
H and W must be divisible by 2**depth (16 for the default depth 4);
``UNetColorizer`` pads arbitrary images automatically.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class DoubleConv(nn.Sequential):
    """(3x3 conv -> BatchNorm -> ReLU) x 2."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 3, base_channels: int = 32, depth: int = 4):
        super().__init__()
        self.depth = depth
        chs = [base_channels * 2**i for i in range(depth + 1)]   # e.g. 32, 64, 128, 256, 512

        self.encoders = nn.ModuleList([DoubleConv(in_channels, chs[0])])
        self.encoders.extend(DoubleConv(chs[i], chs[i + 1]) for i in range(depth))
        self.pool = nn.MaxPool2d(2)

        self.upsamples = nn.ModuleList(nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2) for i in reversed(range(depth)))
        self.decoders = nn.ModuleList(DoubleConv(chs[i] * 2, chs[i]) for i in reversed(range(depth)))
        self.head = nn.Conv2d(chs[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        factor = 2**self.depth
        if x.shape[-2] % factor or x.shape[-1] % factor:
            raise ValueError(f"H and W must be divisible by {factor}, got {tuple(x.shape[-2:])}")

        skips = []
        for i, enc in enumerate(self.encoders):
            x = enc(x if i == 0 else self.pool(x))
            skips.append(x)
        x = skips.pop()                       # bottleneck
        for up, dec in zip(self.upsamples, self.decoders):
            x = dec(torch.cat([up(x), skips.pop()], dim=1))
        return torch.sigmoid(self.head(x))


def build_unet(model_cfg: dict, in_channels: int) -> UNet:
    return UNet(
        in_channels=in_channels,
        out_channels=model_cfg["out_channels"],
        base_channels=model_cfg["base_channels"],
        depth=model_cfg.get("depth", 4),
    )


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[UNet, dict]:
    """Load a checkpoint written by the trainer. Returns ``(model in eval mode, metadata)``."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_unet(ckpt["model_cfg"], in_channels=len(ckpt["inputs"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    return model, meta

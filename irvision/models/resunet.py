"""U-Net with an ImageNet-pretrained ResNet-34 encoder, plus an optional land-cover head.

    RGB     (B, 3, H, W) in [0, 1]      colorization output (sigmoid)
    logits  (B, K, H, W)                land-cover classes (auxiliary head, training signal
                                        and "semantic mask" output)

The single IR channel is repeated to three channels and ImageNet-normalized so the
pretrained encoder sees familiar statistics. H and W must be multiples of 32
(``UNetColorizer`` pads automatically via ``model.depth = 5``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet34_Weights, resnet34

from irvision.models.unet import DoubleConv

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DecoderBlock(nn.Module):
    """Upsample x2, concatenate the skip connection, two 3x3 convs."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResUNet34(nn.Module):
    depth = 5   # total downsampling 2**5 = 32

    def __init__(self, in_channels: int = 1, out_channels: int = 3, num_classes: int = 0, pretrained: bool = True):
        super().__init__()
        enc = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        self.in_channels = in_channels
        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)          # /2,  64 ch
        self.pool = enc.maxpool                                          # /4
        self.layer1, self.layer2, self.layer3, self.layer4 = enc.layer1, enc.layer2, enc.layer3, enc.layer4
        #                                                    /4 64, /8 128, /16 256, /32 512
        self.dec4 = DecoderBlock(512, 256, 256)   # -> /16
        self.dec3 = DecoderBlock(256, 128, 128)   # -> /8
        self.dec2 = DecoderBlock(128, 64, 64)     # -> /4
        self.dec1 = DecoderBlock(64, 64, 64)      # -> /2
        self.dec0 = DecoderBlock(64, 0, 32)       # -> /1
        self.rgb_head = nn.Conv2d(32, out_channels, 1)
        self.num_classes = num_classes
        self.seg_head = nn.Conv2d(32, num_classes, 1) if num_classes else None
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] % 32 or x.shape[-1] % 32:
            raise ValueError(f"H and W must be divisible by 32, got {tuple(x.shape[-2:])}")
        x = x[:, :1].repeat(1, 3, 1, 1) if self.in_channels == 1 else x[:, :3]
        x = (x - self.mean) / self.std
        s1 = self.stem(x)
        s2 = self.layer1(self.pool(s1))
        s3 = self.layer2(s2)
        s4 = self.layer3(s3)
        x = self.layer4(s4)
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        return self.dec0(x, None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Colorized RGB only (what inference uses)."""
        return torch.sigmoid(self.rgb_head(self._features(x)))

    def forward_with_aux(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """RGB and land-cover logits (training with the auxiliary head)."""
        f = self._features(x)
        logits = self.seg_head(f) if self.seg_head is not None else None
        return torch.sigmoid(self.rgb_head(f)), logits

    @torch.no_grad()
    def predict_landcover(self, x: torch.Tensor) -> torch.Tensor | None:
        """(B, H, W) class indices from the auxiliary head, or None if there is no head."""
        _, logits = self.forward_with_aux(x)
        return None if logits is None else logits.argmax(1)

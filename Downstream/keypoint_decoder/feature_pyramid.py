"""A compact UPerNet/ViTDet-style pyramid for same-resolution ViT features."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _norm_groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels, kernel_size,
                padding=kernel_size // 2, bias=False,
            ),
            nn.GroupNorm(_norm_groups(out_channels), out_channels),
            nn.GELU(),
        )


class ViTFeaturePyramid(nn.Module):
    """Fuse four transformer blocks while recovering a stride-four map."""

    def __init__(self, in_channels: int = 768, width: int = 128):
        super().__init__()
        self.width = int(width)
        self.lateral = nn.ModuleList(
            [nn.Conv2d(in_channels, width, 1) for _ in range(4)]
        )
        self.smooth = nn.ModuleList(
            [ConvNormAct(width, width) for _ in range(4)]
        )

    def forward(
        self, features: list[torch.Tensor], output_size: tuple[int, int]
    ) -> torch.Tensor:
        if len(features) != 4:
            raise ValueError(f"Expected four ViT maps, received {len(features)}")
        output_h, output_w = output_size
        # The shallow block supplies fine localization while the deepest block
        # supplies the coarsest semantic map, matching a standard UPerNet FPN.
        sizes = [
            (output_h, output_w),                            # nominal stride 4
            (max(1, output_h // 2), max(1, output_w // 2)),  # nominal stride 8
            (max(1, output_h // 4), max(1, output_w // 4)),  # nominal stride 16
            (max(1, output_h // 8), max(1, output_w // 8)),  # nominal stride 32
        ]
        laterals = [layer(value) for layer, value in zip(self.lateral, features)]
        current = None
        for index in reversed(range(4)):
            value = F.interpolate(
                laterals[index], size=sizes[index], mode="bilinear", align_corners=False
            )
            if current is not None:
                value = value + F.interpolate(
                    current, size=sizes[index], mode="bilinear", align_corners=False
                )
            current = self.smooth[index](value)
        return current

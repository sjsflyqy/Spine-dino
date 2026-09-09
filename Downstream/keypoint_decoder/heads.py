"""Prediction heads and the task-07 mask-gated cross-view connection."""

from __future__ import annotations

import torch
from torch import nn

from .feature_pyramid import ConvNormAct


class StructuredLandmarkHead(nn.Module):
    def __init__(self, channels: int, hidden_channels: int | None = None):
        super().__init__()
        hidden = int(hidden_channels or channels)
        self.heatmap = nn.Sequential(
            ConvNormAct(channels, hidden), nn.Conv2d(hidden, 1, 1)
        )
        self.regression = nn.Sequential(
            ConvNormAct(channels, hidden), nn.Conv2d(hidden, 2, 1)
        )
        # A depthwise 7x7 layer gives the corner head the reference model's
        # wider receptive field without a prohibitively large dense convolution.
        self.corners = nn.Sequential(
            nn.Conv2d(channels, channels, 7, padding=3, groups=channels, bias=False),
            nn.GroupNorm(32 if channels % 32 == 0 else 1, channels),
            nn.GELU(),
            nn.Conv2d(channels, 8, 1),
        )
        self.segmentation = nn.Sequential(
            ConvNormAct(channels, hidden), nn.Conv2d(hidden, 1, 1)
        )
        nn.init.constant_(self.heatmap[-1].bias, -2.19)

    def forward(self, feature: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "hm": self.heatmap(feature),
            "reg": self.regression(feature),
            "corner_offsets": self.corners(feature),
            "segmentation": self.segmentation(feature),
        }


class MaskGatedViewFusion(nn.Module):
    """Fuse only the deepest AP/LAT maps using learned coarse spine masks."""

    def __init__(self, channels: int = 768):
        super().__init__()
        hidden = max(channels // 4, 64)
        self.mask_head = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        self.project = nn.Conv2d(2 * channels, channels, 1)

    def forward(
        self, ap: torch.Tensor, lat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ap_mask = self.mask_head(ap)
        lat_mask = self.mask_head(lat)
        shared = ap * ap_mask.sigmoid() + lat * lat_mask.sigmoid()
        return (
            self.project(torch.cat((ap, shared), dim=1)),
            self.project(torch.cat((lat, shared), dim=1)),
            ap_mask,
            lat_mask,
        )

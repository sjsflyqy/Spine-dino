from __future__ import annotations

import torch
from torch import nn

from backbone import build_backbone


class LumbarKeypointLinearProbe(nn.Module):
    """Frozen visual tokens plus one affine heatmap predictor."""

    def __init__(self, backbone_name: str, weights: str | None, landmarks: int = 22,
                 adapter: dict | None = None):
        super().__init__()
        self.backbone = build_backbone(backbone_name, weights, freeze=True, adapter=adapter)
        self.head = nn.Conv2d(self.backbone.embedding_dim, landmarks, kernel_size=1)

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.backbone.adapter_enabled:
            self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor):
        context = (
            torch.enable_grad()
            if self.training and self.backbone.adapter_enabled
            else torch.no_grad()
        )
        with context:
            features = self.backbone(images).feature_map
        return self.head(features)

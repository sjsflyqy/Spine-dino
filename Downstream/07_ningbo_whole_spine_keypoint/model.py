from __future__ import annotations

import torch
from torch import nn

from backbone import build_backbone


class PairedKeypointLinearProbe(nn.Module):
    """Paired AP/LAT frozen features with affine 1x1 landmark heads."""

    def __init__(self, backbone_name: str, weights: str | None, landmarks: int = 68,
                 adapter: dict | None = None):
        super().__init__()
        self.backbone = build_backbone(backbone_name, weights, freeze=True, adapter=adapter)
        channels = 2 * self.backbone.embedding_dim
        self.ap_head = nn.Conv2d(channels, landmarks, kernel_size=1)
        self.lat_head = nn.Conv2d(channels, landmarks, kernel_size=1)

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.backbone.adapter_enabled:
            self.backbone.eval()
        return self

    def forward(self, ap_images: torch.Tensor, lat_images: torch.Tensor):
        batch = ap_images.shape[0]
        context = (
            torch.enable_grad()
            if self.training and self.backbone.adapter_enabled
            else torch.no_grad()
        )
        with context:
            features = self.backbone(torch.cat((ap_images, lat_images), dim=0)).feature_map
        ap_features, lat_features = features[:batch], features[batch:]
        return {
            "ap": self.ap_head(torch.cat((ap_features, lat_features), dim=1)),
            "lat": self.lat_head(torch.cat((lat_features, ap_features), dim=1)),
        }

    def probe_state_dict(self):
        return {"ap_head": self.ap_head.state_dict(), "lat_head": self.lat_head.state_dict()}

    def load_probe_state_dict(self, state):
        self.ap_head.load_state_dict(state["ap_head"])
        self.lat_head.load_state_dict(state["lat_head"])

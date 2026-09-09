"""Structured decoder for the six lumbar/sacral landmark objects in task 08."""

from __future__ import annotations

import torch
from torch import nn

from backbone import build_backbone
from keypoint_decoder import (
    StructuredLandmarkHead,
    ViTFeaturePyramid,
    extract_intermediate_feature_maps,
)


class LumbarStructuredDecoder(nn.Module):
    """Multi-block ViT pyramid with center, corner-offset, and mask heads."""

    def __init__(
        self,
        backbone_name: str,
        weights: str | None,
        decoder: dict | None = None,
        adapter: dict | None = None,
    ) -> None:
        super().__init__()
        decoder = decoder or {}
        self.backbone = build_backbone(
            backbone_name, weights, freeze=True, adapter=adapter
        )
        self.intermediate_layers = tuple(
            int(index)
            for index in decoder.get("intermediate_layers", (2, 5, 8, 11))
        )
        self.output_stride = int(decoder.get("output_stride", 4))
        if self.output_stride < 1:
            raise ValueError("decoder.output_stride must be >= 1")
        width = int(decoder.get("width", 128))
        self.decoder = ViTFeaturePyramid(self.backbone.embedding_dim, width)
        self.head = StructuredLandmarkHead(
            width, int(decoder.get("head_width", width))
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.backbone.adapter_enabled:
            self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        context = (
            torch.enable_grad()
            if self.training and self.backbone.adapter_enabled
            else torch.no_grad()
        )
        with context:
            features = extract_intermediate_feature_maps(
                self.backbone, images, self.intermediate_layers
            )
        output_size = (
            max(1, images.shape[-2] // self.output_stride),
            max(1, images.shape[-1] // self.output_stride),
        )
        return self.head(self.decoder(features, output_size))

"""Paired AP/LAT structured decoder for task 07."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from backbone import build_backbone
from keypoint_decoder import (
    MaskGatedViewFusion,
    StructuredLandmarkHead,
    ViTFeaturePyramid,
    extract_intermediate_feature_maps,
)


class PairedStructuredDecoder(nn.Module):
    """Shared multi-scale decoder with deepest-level mask-gated view fusion."""

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
        self.view_fusion = MaskGatedViewFusion(self.backbone.embedding_dim)
        self.decoder = ViTFeaturePyramid(self.backbone.embedding_dim, width)
        self.head = StructuredLandmarkHead(
            width, int(decoder.get("head_width", width))
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.backbone.adapter_enabled:
            self.backbone.eval()
        return self

    def forward(
        self, ap_images: torch.Tensor, lat_images: torch.Tensor
    ) -> dict[str, dict[str, torch.Tensor]]:
        if ap_images.shape != lat_images.shape:
            raise ValueError("Task 07 AP and LAT batches must have identical shapes")
        batch_size = ap_images.shape[0]
        combined = torch.cat((ap_images, lat_images), dim=0)
        context = (
            torch.enable_grad()
            if self.training and self.backbone.adapter_enabled
            else torch.no_grad()
        )
        with context:
            combined_features = extract_intermediate_feature_maps(
                self.backbone, combined, self.intermediate_layers
            )
        ap_features = [value[:batch_size] for value in combined_features]
        lat_features = [value[batch_size:] for value in combined_features]
        (
            ap_features[-1],
            lat_features[-1],
            ap_coarse_mask,
            lat_coarse_mask,
        ) = self.view_fusion(ap_features[-1], lat_features[-1])
        output_size = (
            max(1, ap_images.shape[-2] // self.output_stride),
            max(1, ap_images.shape[-1] // self.output_stride),
        )
        ap_output = self.head(self.decoder(ap_features, output_size))
        lat_output = self.head(self.decoder(lat_features, output_size))
        ap_output["coarse_segmentation"] = F.interpolate(
            ap_coarse_mask, output_size, mode="bilinear", align_corners=False
        )
        lat_output["coarse_segmentation"] = F.interpolate(
            lat_coarse_mask, output_size, mode="bilinear", align_corners=False
        )
        return {"ap": ap_output, "lat": lat_output}

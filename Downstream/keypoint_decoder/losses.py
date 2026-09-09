"""Losses for center heatmaps, offsets, and auxiliary vertebral masks."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _gather_map(feature: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    flattened = feature.flatten(2).transpose(1, 2)
    return flattened.gather(
        1, indices.unsqueeze(-1).expand(-1, -1, flattened.shape[-1])
    )


def centernet_focal_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = logits.sigmoid().clamp(1e-6, 1.0 - 1e-6)
    positive = target.eq(1).to(logits.dtype)
    negative = target.lt(1).to(logits.dtype)
    negative_weight = (1.0 - target).pow(4)
    positive_loss = -(prediction.log() * (1.0 - prediction).pow(2) * positive).sum()
    negative_loss = -(
        (1.0 - prediction).log()
        * prediction.pow(2)
        * negative_weight
        * negative
    ).sum()
    return (positive_loss + negative_loss) / positive.sum().clamp_min(1.0)


def _segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if logits.shape[-2:] != target.shape[-2:]:
        logits = F.interpolate(logits, target.shape[-2:], mode="bilinear", align_corners=False)
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probability = logits.sigmoid()
    intersection = (probability * target).sum(dim=(-2, -1))
    denominator = probability.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return bce + dice


def structured_landmark_loss(
    output: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    weights: dict | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = weights or {}
    indices = target["indices"]
    predicted_reg = _gather_map(output["reg"], indices)
    object_mask = target["object_mask"].unsqueeze(-1).expand_as(
        predicted_reg
    ).to(output["reg"].dtype)
    reg_loss = F.l1_loss(
        predicted_reg * object_mask, target["reg"] * object_mask, reduction="sum"
    ) / object_mask.sum().clamp_min(1.0)
    predicted_corners = _gather_map(output["corner_offsets"], indices)
    corner_mask = target["corner_mask"].to(predicted_corners.dtype)
    corner_loss = F.l1_loss(
        predicted_corners * corner_mask,
        target["corner_offsets"] * corner_mask,
        reduction="sum",
    ) / corner_mask.sum().clamp_min(1.0)
    heatmap_loss = centernet_focal_loss(output["hm"], target["hm"])
    segmentation_loss = _segmentation_loss(
        output["segmentation"], target["segmentation"]
    )
    coarse_loss = output["hm"].new_zeros(())
    if "coarse_segmentation" in output:
        coarse_loss = _segmentation_loss(
            output["coarse_segmentation"], target["segmentation"]
        )
    parts = {
        "heatmap": heatmap_loss,
        "center_offset": reg_loss,
        "corner_offset": corner_loss,
        "segmentation": segmentation_loss,
        "coarse_segmentation": coarse_loss,
    }
    total = (
        float(weights.get("heatmap", 1.0)) * heatmap_loss
        + float(weights.get("center_offset", 5.0)) * reg_loss
        + float(weights.get("corner_offset", 1.0)) * corner_loss
        + float(weights.get("segmentation", 1.0)) * segmentation_loss
        + float(weights.get("coarse_segmentation", 0.5)) * coarse_loss
    )
    return total, parts

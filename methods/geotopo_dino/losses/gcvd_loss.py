"""Dense and region-aligned losses from the GeoTopo-DINO design."""

from __future__ import annotations

import torch
from torch import nn


def valid_mean_pool(tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    if tokens.shape[:2] != valid_mask.shape:
        raise ValueError("token and valid-mask shapes do not match")
    weights = valid_mask.to(dtype=tokens.dtype).unsqueeze(-1)
    return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class GeometryDistillationLoss(nn.Module):
    """Cosine loss over aligned patches and their valid-aware pooled regions."""

    def forward(
        self,
        *,
        student_patch_embeddings: torch.Tensor,
        teacher_patch_embeddings: torch.Tensor,
        student_region_embeddings: torch.Tensor,
        teacher_region_embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if student_patch_embeddings.shape != teacher_patch_embeddings.shape:
            raise ValueError("student and teacher patch embeddings must match")
        if student_patch_embeddings.shape[:2] != valid_mask.shape:
            raise ValueError("valid mask does not match patch embeddings")

        patch_distance = 1.0 - (student_patch_embeddings.float() * teacher_patch_embeddings.float()).sum(-1)
        per_crop = (patch_distance * valid_mask.float()).sum(-1) / valid_mask.sum(-1).clamp_min(1)
        valid_crops = valid_mask.any(dim=-1)
        dense = per_crop[valid_crops].mean() if valid_crops.any() else patch_distance.sum() * 0.0

        region_distance = 1.0 - (
            student_region_embeddings.float() * teacher_region_embeddings.float()
        ).sum(-1)
        region = region_distance[valid_crops].mean() if valid_crops.any() else region_distance.sum() * 0.0
        return {"dense": dense, "region": region}


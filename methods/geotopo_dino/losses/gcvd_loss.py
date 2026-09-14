"""Dense and region-aligned losses from the GeoTopo-DINO design."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


def valid_mean_pool(tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    if tokens.shape[:2] != valid_mask.shape:
        raise ValueError("token and valid-mask shapes do not match")
    weights = valid_mask.to(dtype=tokens.dtype).unsqueeze(-1)
    return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class GeometryDistillationLoss(nn.Module):
    """Cosine loss over aligned patches and their valid-aware pooled regions."""

    def dense_loss(
        self,
        student_patch_embeddings: torch.Tensor,
        teacher_patch_embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if student_patch_embeddings.shape != teacher_patch_embeddings.shape:
            raise ValueError("student and teacher patch embeddings must match")
        if student_patch_embeddings.shape[:2] != valid_mask.shape:
            raise ValueError("valid mask does not match patch embeddings")
        patch_distance = 1.0 - (
            student_patch_embeddings.float() * teacher_patch_embeddings.float()
        ).sum(-1)
        per_crop = (patch_distance * valid_mask.float()).sum(-1) / valid_mask.sum(-1).clamp_min(1)
        valid_crops = valid_mask.any(dim=-1)
        return per_crop[valid_crops].mean() if valid_crops.any() else patch_distance.sum() * 0.0

    def region_loss(
        self,
        student_region_embeddings: torch.Tensor,
        teacher_region_embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if student_region_embeddings.shape != teacher_region_embeddings.shape:
            raise ValueError("student and teacher region embeddings must match")
        if student_region_embeddings.shape[0] != valid_mask.shape[0]:
            raise ValueError("valid mask does not match region embeddings")
        valid_crops = valid_mask.any(dim=-1)
        region_distance = 1.0 - (
            student_region_embeddings.float() * teacher_region_embeddings.float()
        ).sum(-1)
        return (
            region_distance[valid_crops].mean()
            if valid_crops.any()
            else region_distance.sum() * 0.0
        )

    def forward(
        self,
        *,
        student_patch_embeddings: torch.Tensor,
        teacher_patch_embeddings: torch.Tensor,
        student_region_embeddings: torch.Tensor,
        teacher_region_embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        dense = self.dense_loss(
            student_patch_embeddings,
            teacher_patch_embeddings,
            valid_mask,
        )
        region = self.region_loss(
            student_region_embeddings,
            teacher_region_embeddings,
            valid_mask,
        )
        return {"dense": dense, "region": region}


def compute_gcvd_warmup_scale(
    *,
    warmup_type: str,
    iteration: int,
    schedule_max_iterations: int,
    warmup_fraction: float,
    warmup_iterations: int,
) -> float:
    """Return a resume-stable GCVD scale from the global optimizer iteration."""
    aliases = {
        "fraction_warmup": "fraction",
        "fixed_iter_warmup": "fixed_iter",
    }
    normalized_type = aliases.get(str(warmup_type), str(warmup_type))
    if iteration < 0:
        raise ValueError("iteration must be non-negative")
    if normalized_type == "no_warmup":
        return 1.0
    if normalized_type == "fraction":
        if not 0.0 <= warmup_fraction <= 1.0:
            raise ValueError("gcvd.warmup_fraction must be in [0, 1]")
        if warmup_fraction == 0.0:
            return 1.0
        denominator = max(float(schedule_max_iterations) * warmup_fraction, 1.0)
        return min(float(iteration) / denominator, 1.0)
    if normalized_type == "fixed_iter":
        if warmup_iterations < 0:
            raise ValueError("gcvd.warmup_iterations must be non-negative")
        if warmup_iterations == 0:
            return 1.0
        return min(float(iteration) / float(warmup_iterations), 1.0)
    raise ValueError(
        "gcvd.warmup_type must be one of no_warmup, fraction, or fixed_iter"
    )


class GCVDPrototypeLoss(nn.Module):
    """Centered teacher-student prototype cross entropy for valid GCVD patches."""

    def __init__(
        self,
        out_dim: int,
        *,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ) -> None:
        super().__init__()
        if out_dim <= 0:
            raise ValueError("GCVD prototype count must be positive")
        if teacher_temp <= 0 or student_temp <= 0:
            raise ValueError("GCVD teacher/student temperatures must be positive")
        if not 0.0 <= center_momentum < 1.0:
            raise ValueError("gcvd.center_momentum must be in [0, 1)")
        self.out_dim = int(out_dim)
        self.teacher_temp = float(teacher_temp)
        self.student_temp = float(student_temp)
        self.center_momentum = float(center_momentum)
        # This buffer belongs to the top-level model state, so FSDP checkpoints
        # restore it together with both GCVD heads and the optimizer state.
        self.register_buffer("center", torch.zeros(1, self.out_dim))

    @staticmethod
    def _distribution_diagnostics(
        probabilities: torch.Tensor,
        *,
        include_usage: bool,
    ) -> dict[str, torch.Tensor]:
        if probabilities.shape[0] == 0:
            zero = probabilities.sum()
            diagnostics = {"entropy": zero, "max_prob": zero}
            if include_usage:
                usage = torch.zeros(
                    probabilities.shape[-1],
                    dtype=torch.long,
                    device=probabilities.device,
                )
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(usage)
                diagnostics["active_prototype_ratio"] = usage.gt(0).float().mean()
            return diagnostics
        probabilities = probabilities.float()
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1).mean()
        max_prob = probabilities.max(dim=-1).values.mean()
        diagnostics = {
            "entropy": entropy,
            "max_prob": max_prob,
        }
        if include_usage:
            assignments = probabilities.argmax(dim=-1)
            usage = torch.bincount(assignments, minlength=probabilities.shape[-1])
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(usage)
            diagnostics["active_prototype_ratio"] = usage.gt(0).float().mean()
        return diagnostics

    @torch.no_grad()
    def teacher_probabilities(
        self,
        teacher_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if teacher_logits.ndim != 2 or teacher_logits.shape[-1] != self.out_dim:
            raise ValueError("teacher GCVD logits must have shape [valid_patches, prototypes]")
        logits = teacher_logits.float()
        probabilities = F.softmax((logits - self.center) / self.teacher_temp, dim=-1)
        diagnostics = self._distribution_diagnostics(probabilities, include_usage=True)
        self.update_center(logits)
        return probabilities.detach(), diagnostics

    @torch.no_grad()
    def update_center(self, teacher_logits: torch.Tensor) -> None:
        local_sum = teacher_logits.float().sum(dim=0, keepdim=True)
        local_count = torch.tensor(
            [teacher_logits.shape[0]],
            dtype=torch.float32,
            device=teacher_logits.device,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local_sum)
            dist.all_reduce(local_count)
        if local_count.item() == 0:
            return
        batch_center = local_sum / local_count
        self.center.mul_(self.center_momentum).add_(
            batch_center,
            alpha=1.0 - self.center_momentum,
        )

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_probabilities: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if student_logits.shape != teacher_probabilities.shape:
            raise ValueError("student logits and teacher probabilities must have identical shapes")
        if student_logits.ndim != 2 or student_logits.shape[-1] != self.out_dim:
            raise ValueError("student GCVD logits must have shape [valid_patches, prototypes]")
        if student_logits.shape[0] == 0:
            zero = student_logits.sum() * 0.0
            # Keep the usage all-reduce collective aligned across ranks even if
            # only some ranks have no valid geometric correspondences.
            diagnostics = self._distribution_diagnostics(
                student_logits.detach().float(),
                include_usage=True,
            )
            return zero, diagnostics

        student_log_probabilities = F.log_softmax(
            student_logits.float() / self.student_temp,
            dim=-1,
        )
        loss = -(teacher_probabilities.float() * student_log_probabilities).sum(-1).mean()
        with torch.no_grad():
            student_probabilities = student_log_probabilities.detach().exp()
            diagnostics = self._distribution_diagnostics(
                student_probabilities,
                include_usage=True,
            )
        return loss, diagnostics

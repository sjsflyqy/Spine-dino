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
    """Balanced teacher-student prototype cross entropy for valid GCVD patches."""

    def __init__(
        self,
        out_dim: int,
        *,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
        teacher_centering: str = "centering",
        sinkhorn_iterations: int = 3,
    ) -> None:
        super().__init__()
        if out_dim <= 0:
            raise ValueError("GCVD prototype count must be positive")
        if teacher_temp <= 0 or student_temp <= 0:
            raise ValueError("GCVD teacher/student temperatures must be positive")
        if not 0.0 <= center_momentum < 1.0:
            raise ValueError("gcvd.center_momentum must be in [0, 1)")
        if teacher_centering not in ("centering", "sinkhorn_knopp"):
            raise ValueError(
                "gcvd.teacher_centering must be centering or sinkhorn_knopp"
            )
        if sinkhorn_iterations <= 0:
            raise ValueError("gcvd.sinkhorn_iterations must be positive")
        self.out_dim = int(out_dim)
        self.teacher_temp = float(teacher_temp)
        self.student_temp = float(student_temp)
        self.center_momentum = float(center_momentum)
        self.teacher_centering = str(teacher_centering)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        # This buffer belongs to the top-level model state, so FSDP checkpoints
        # restore it together with both GCVD heads and the optimizer state.
        self.register_buffer("center", torch.zeros(1, self.out_dim))

    @staticmethod
    def _distribution_diagnostics(
        probabilities: torch.Tensor,
        *,
        include_usage: bool,
    ) -> dict[str, torch.Tensor]:
        probabilities = probabilities.float()
        if probabilities.shape[0] == 0:
            zero = probabilities.sum()
            diagnostics = {"entropy": zero, "max_prob": zero}
        else:
            entropy = -(
                probabilities * probabilities.clamp_min(1e-12).log()
            ).sum(-1).mean()
            max_prob = probabilities.max(dim=-1).values.mean()
            diagnostics = {
                "entropy": entropy,
                "max_prob": max_prob,
            }
        if include_usage:
            if probabilities.shape[0] == 0:
                usage = torch.zeros(
                    probabilities.shape[-1],
                    dtype=torch.long,
                    device=probabilities.device,
                )
            else:
                assignments = probabilities.argmax(dim=-1)
                usage = torch.bincount(assignments, minlength=probabilities.shape[-1])
            probability_sum = probabilities.sum(dim=0)
            probability_count = torch.tensor(
                [probabilities.shape[0]],
                dtype=torch.float32,
                device=probabilities.device,
            )
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(usage)
                dist.all_reduce(probability_sum)
                dist.all_reduce(probability_count)
            diagnostics["active_prototype_ratio"] = usage.gt(0).float().mean()
            if probability_count.item() == 0:
                marginal = probability_sum
            else:
                marginal = probability_sum / probability_count
            marginal_entropy = -(
                marginal * marginal.clamp_min(1e-12).log()
            ).sum()
            diagnostics["marginal_entropy"] = marginal_entropy
            diagnostics["effective_prototype_ratio"] = (
                marginal_entropy.exp() / probabilities.shape[-1]
            )
        return diagnostics

    @torch.no_grad()
    def teacher_probabilities(
        self,
        teacher_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if teacher_logits.ndim != 2 or teacher_logits.shape[-1] != self.out_dim:
            raise ValueError("teacher GCVD logits must have shape [valid_patches, prototypes]")
        logits = teacher_logits.float()
        if self.teacher_centering == "centering":
            probabilities = F.softmax((logits - self.center) / self.teacher_temp, dim=-1)
            self.update_center(logits)
        else:
            probabilities = self.sinkhorn_knopp_teacher(logits)
        diagnostics = self._distribution_diagnostics(probabilities, include_usage=True)
        return probabilities.detach(), diagnostics

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_logits: torch.Tensor) -> torch.Tensor:
        """Return globally balanced assignments for a variable number of valid tokens."""
        logits = teacher_logits.float()
        local_count = torch.tensor(
            [logits.shape[0]], dtype=torch.long, device=logits.device
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local_count)
        global_count = int(local_count.item())
        if global_count == 0:
            return logits

        # Per-sample shifts are absorbed by Sinkhorn's column normalization and
        # prevent a low-temperature teacher from overflowing or producing an
        # all-zero column.
        scaled_logits = logits / self.teacher_temp
        if logits.shape[0] > 0:
            scaled_logits = scaled_logits - scaled_logits.max(dim=1, keepdim=True).values
        assignments = torch.exp(scaled_logits).clamp_min(
            torch.finfo(scaled_logits.dtype).tiny
        ).t()
        total_mass = assignments.sum()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(total_mass)
        assignments /= total_mass.clamp_min(torch.finfo(assignments.dtype).tiny)

        for _ in range(self.sinkhorn_iterations):
            prototype_mass = assignments.sum(dim=1, keepdim=True)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(prototype_mass)
            assignments /= prototype_mass.clamp_min(
                torch.finfo(assignments.dtype).tiny
            )
            assignments /= self.out_dim

            sample_mass = assignments.sum(dim=0, keepdim=True)
            assignments /= sample_mass.clamp_min(torch.finfo(assignments.dtype).tiny)
            assignments /= global_count

        # Columns are probability distributions for local valid tokens.
        return (assignments * global_count).t()

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

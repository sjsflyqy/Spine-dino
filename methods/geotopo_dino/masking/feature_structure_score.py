"""Local teacher-token variation, without anatomy labels or mask selection.

Shared by the offline diagnostic and the online highest-A mask policy.
Padding is excluded from differences, quantiles, and smoothing, not just display.
"""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class StructureScore:
    difference: torch.Tensor
    normalized: torch.Tensor
    score: torch.Tensor
    neighbor_count: torch.Tensor
    metrics: dict


@torch.no_grad()
def compute_structure_score(
    patch_tokens: torch.Tensor,
    valid: torch.Tensor,
    *,
    radius: int = 1,
    smooth_kernel: int = 3,
    quantiles: tuple[float, float] = (0.05, 0.95),
    min_spread: float = 1e-5,
) -> StructureScore:
    """D_i = mean valid-neighbor (1-cos); S = valid-aware smooth(Norm(D)).

    Accept [H*W,C] or [H,W,C] tokens and [H,W] validity. A square radius
    neighborhood excludes its center. Unsupported patches return zero; a flat
    map is not stretched into apparent structure. All outputs are detached FP32.
    """
    if valid.ndim != 2 or min(valid.shape) < 1:
        raise ValueError("valid must be a nonempty [H,W] mask")
    if radius < 1 or not isinstance(radius, int):
        raise ValueError("radius must be a positive integer")
    if smooth_kernel < 1 or smooth_kernel % 2 != 1:
        raise ValueError("smooth_kernel must be a positive odd integer")
    lo, hi = quantiles
    if not 0 <= lo < hi <= 1 or not math.isfinite(min_spread) or min_spread <= 0:
        raise ValueError("invalid quantiles or min_spread")
    h, w = valid.shape
    if patch_tokens.ndim == 2 and patch_tokens.shape[0] == h * w:
        patch_tokens = patch_tokens.reshape(h, w, -1)
    if patch_tokens.ndim != 3 or patch_tokens.shape[:2] != (h, w) or patch_tokens.shape[-1] < 1:
        raise ValueError("tokens must be [H*W,C] or [H,W,C]")
    if patch_tokens.device != valid.device:
        raise ValueError("tokens and validity must be on the same device")
    valid = valid.bool()
    tokens = patch_tokens.detach().float()
    if not torch.isfinite(tokens[valid]).all():
        raise ValueError("valid tokens must be finite")
    if (tokens[valid].norm(dim=-1) < 1e-8).any():
        raise ValueError("valid tokens must have nonzero norm")
    tokens = F.normalize(torch.where(valid[..., None], tokens, 0), dim=-1)
    total = tokens.new_zeros(h, w)
    count = tokens.new_zeros(h, w)
    for dy in range(-min(radius, h - 1), min(radius, h - 1) + 1):
        for dx in range(-min(radius, w - 1), min(radius, w - 1) + 1):
            if dy == 0 and dx == 0:
                continue
            y0, y1 = max(0, -dy), min(h, h - dy)
            x0, x1 = max(0, -dx), min(w, w - dx)
            center = (slice(y0, y1), slice(x0, x1))
            neighbor = (slice(y0 + dy, y1 + dy), slice(x0 + dx, x1 + dx))
            pair_valid = valid[center] & valid[neighbor]
            distance = (1 - (tokens[center] * tokens[neighbor]).sum(-1)).clamp(0, 2)
            total[center] += distance * pair_valid
            count[center] += pair_valid
    supported = valid & (count > 0)
    difference = total / count.clamp_min(1)
    normalized = torch.zeros_like(difference)
    values = difference[supported]
    lower = upper = spread = 0.0
    near_constant = True
    if values.numel():
        bounds = torch.quantile(values, values.new_tensor([lo, hi]))
        lower, upper = map(float, bounds)
        spread = upper - lower
        near_constant = spread < min_spread
        if not near_constant:
            normalized = ((difference - lower) / spread).clamp(0, 1) * supported
    # Divide by the actual support, so padding cannot dilute edge scores.
    pooled = F.avg_pool2d(normalized[None, None], smooth_kernel, 1, smooth_kernel // 2)[0, 0]
    support = F.avg_pool2d(supported.float()[None, None], smooth_kernel, 1, smooth_kernel // 2)[0, 0]
    score = pooled / support.clamp_min(1e-8) * supported
    interior = F.avg_pool2d(valid.float()[None, None], 3, 1, 1)[0, 0] >= 1 - 1e-6
    border = valid & ~interior
    mass = score.sum()
    metrics = {
        "valid_count": int(valid.sum()), "supported_count": int(supported.sum()),
        "raw_mean": float(values.mean()) if values.numel() else 0.0,
        "raw_std": float(values.std(unbiased=False)) if values.numel() else 0.0,
        "raw_min": float(values.min()) if values.numel() else 0.0,
        "raw_max": float(values.max()) if values.numel() else 0.0,
        "normalization_low": lower, "normalization_high": upper,
        "raw_quantile_spread": spread, "near_constant": near_constant,
        "score_mean": float(score[supported].mean()) if values.numel() else 0.0,
        "border_score_mass_fraction": float(score[border].sum() / mass) if mass > 0 else 0.0,
    }
    return StructureScore(difference, normalized, score, count, metrics)

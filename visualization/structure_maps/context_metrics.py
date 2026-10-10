"""Offline context diagnostics; no training state, loss, or selection is changed."""

import numpy as np
import torch
from torch.nn import functional as F

from methods.geotopo_dino.losses.wavelet_reconstruction_loss import haar_details


def _check_masks(target, input_mask, valid):
    if target.ndim != 2 or target.shape != input_mask.shape or target.shape != valid.shape:
        raise ValueError("expected matching 2D masks")
    if any(mask.dtype != torch.bool for mask in (target, input_mask, valid)):
        raise ValueError("masks must be boolean")
    if not target.any() or (input_mask & ~valid).any() or (target & ~input_mask).any():
        raise ValueError("target must be nonempty and contained in the valid input mask")


@torch.no_grad()
def context_score(tokens, target, input_mask, valid, radius=5, threshold=0.7):
    """Maximum cosine to a valid visible token in a Chebyshev neighborhood.

    Tokens come from the unmasked Teacher. Missing neighborhoods remain NaN;
    a conservative aggregate maps them to zero, reported alongside coverage.
    Full-image Teacher attention makes this a heuristic, not a leakage-free
    prediction from visible pixels.
    """
    _check_masks(target, input_mask, valid)
    if radius < 1 or not -1 <= threshold <= 1:
        raise ValueError("invalid radius or similarity threshold")
    h, w = valid.shape
    tokens = tokens.reshape(h * w, -1).detach().float().cpu()
    if not torch.isfinite(tokens).all():
        raise ValueError("nonfinite Teacher features")
    target, input_mask, valid = [mask.cpu() for mask in (target, input_mask, valid)]
    positions = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij"), -1).reshape(-1, 2)
    ids = target.flatten().nonzero().flatten()
    visible = (valid & ~input_mask).flatten().nonzero().flatten()
    values = torch.full((h * w,), torch.nan)
    normalized = F.normalize(tokens, dim=-1)
    if visible.numel():
        nearby = (positions[ids, None] - positions[visible][None]).abs().amax(-1) <= radius
        cosine = normalized[ids] @ normalized[visible].T
        maxima = cosine.masked_fill(~nearby, -torch.inf).amax(-1)
        values[ids] = torch.where(nearby.any(-1), maxima.clamp(-1, 1), torch.nan)
    selected = values[ids]
    supported = torch.isfinite(selected)
    metrics = {
        "P": float(torch.nan_to_num(selected, nan=0.0).mean()),
        "P_supported": float(selected[supported].mean()) if supported.any() else None,
        "missing_fraction": float((~supported).float().mean()),
        "H": float(((~supported) | (selected < threshold)).float().mean()),
        "radius": radius, "threshold": threshold,
    }
    return values.reshape(h, w), metrics


def context_interventions(target, valid, near_radius=2, far_distance=6, repeats=3, seed=0):
    """Remove nearby visible information vs equally many distant visible tokens.

    All outputs contain the original target. If distant capacity is insufficient,
    uniformly subsample the near ring and report the cap explicitly.
    """
    _check_masks(target, target, valid)
    if near_radius < 1 or far_distance <= near_radius or repeats < 1:
        raise ValueError("need positive near radius, greater far distance, and repeats")
    h, w = target.shape
    positions = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij"), -1).reshape(-1, 2)
    target, valid = target.cpu(), valid.cpu()
    distance = (positions[:, None] - positions[target.flatten()][None]).abs().amax(-1).amin(-1).reshape(h, w)
    near_pool = valid & ~target & (distance <= near_radius)
    far_pool = valid & ~target & (distance >= far_distance)
    count = min(int(near_pool.sum()), int(far_pool.sum()))
    if count == 0:
        raise ValueError("no matched near/far intervention is feasible")
    rng = torch.Generator().manual_seed(seed)
    def choose(pool):
        ids = pool.flatten().nonzero().flatten()
        result = torch.zeros_like(pool).flatten()
        result[ids[torch.randperm(len(ids), generator=rng)[:count]]] = True
        return result.reshape(h, w)
    near = near_pool.clone() if int(near_pool.sum()) == count else choose(near_pool)
    far = [choose(far_pool) for _ in range(repeats)]
    metadata = {"extra_count": count, "near_pool_count": int(near_pool.sum()),
                "far_pool_count": int(far_pool.sum()), "near_capped": count < int(near_pool.sum()),
                "near_radius": near_radius, "far_distance": far_distance}
    return target | near, [target | mask for mask in far], metadata


def patch_errors(prediction, target, patch_size=14):
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("expected matching N x flattened-pixel arrays")
    prediction, target = prediction.float(), target.float()
    channels = prediction.shape[-1] // (patch_size ** 2)
    def spatial(values):
        return values.reshape(-1, patch_size, patch_size, channels).permute(0, 3, 1, 2)
    return {"pixel_mse": (prediction - target).square().mean(-1),
            "haar_l1": (haar_details(spatial(prediction)) - haar_details(spatial(target))).abs().mean((1, 2, 3, 4))}


def prototype_errors(student_logits, teacher_probabilities, student_temp=0.1):
    """Fixed centered Teacher targets; KL removes target entropy from CE."""
    if student_logits.shape != teacher_probabilities.shape:
        raise ValueError("prototype shapes differ")
    logp = F.log_softmax(student_logits.float() / student_temp, dim=-1)
    q = teacher_probabilities.float()
    entropy = -(q * q.clamp_min(1e-30).log()).sum(-1)
    ce = -(q * logp).sum(-1)
    return {"ibot_ce": ce, "ibot_kl": ce - entropy, "teacher_entropy": entropy}


def visible_baselines(patches, target, input_mask, valid):
    """Mean visible patch and nearest visible patch; never reads hidden targets."""
    _check_masks(target, input_mask, valid)
    h, w = valid.shape
    ids = target.flatten().nonzero().flatten()
    visible = (valid & ~input_mask).flatten().nonzero().flatten()
    if not len(visible):
        raise ValueError("no visible patch for baselines")
    positions = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij"), -1).reshape(-1, 2)
    nearest = torch.cdist(positions[ids].float(), positions[visible].float()).argmin(-1)
    return {"visible_mean": patches[visible].mean(0).expand(len(ids), -1),
            "nearest_copy": patches[visible[nearest]]}


def rank_values(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2
        start = stop
    return ranks


def spearman(x, y, controls=None):
    x, y = np.asarray(x, float), np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    if controls is not None:
        controls = np.asarray(controls, float)
        valid &= np.isfinite(controls).all(1)
    x, y = rank_values(x[valid]), rank_values(y[valid])
    if len(x) < 3:
        return None
    if controls is not None:
        design = np.column_stack([np.ones(len(x))] + [rank_values(column) for column in controls[valid].T])
        if len(x) <= np.linalg.matrix_rank(design) + 1:
            return None
        x -= design @ np.linalg.lstsq(design, x, rcond=None)[0]
        y -= design @ np.linalg.lstsq(design, y, rcond=None)[0]
    if x.std() < 1e-10 or y.std() < 1e-10:
        return None
    return float(np.corrcoef(x, y)[0, 1])

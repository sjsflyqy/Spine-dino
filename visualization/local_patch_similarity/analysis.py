"""Backbone inference, patch comparisons, and padding-aware correspondence checks."""

import math

import numpy as np
import torch
import torch.nn.functional as F

from methods.geotopo_dino.geometry.transforms import transform_points
from methods.geotopo_dino.geometry.warp import warp_anchor_features_to_local
from .views import patch_centres


@torch.inference_mode()
def extract_local_features(loaded, images, device, batch_size=2):
    if loaded.patch_size != 14 or loaded.generation != 2:
        raise ValueError("This analysis requires native DINOv2/RAD-DINO/MAIRA2 ViT-B/14 weights")
    local = torch.cat([
        loaded.model.forward_features(batch.to(device))["x_norm_patchtokens"].float().cpu()
        for batch in images.split(batch_size)
    ])
    if not torch.isfinite(local).all():
        raise ValueError("Backbone produced non-finite local features")
    return local


@torch.inference_mode()
def extract_features(loaded, views, device, batch_size=2):
    local = extract_local_features(loaded, views.locals, device, batch_size)
    anchor = loaded.model.forward_features(views.anchor[None].to(device))["x_norm_patchtokens"].float().cpu()
    if not torch.isfinite(anchor).all():
        raise ValueError("Backbone produced non-finite features")
    return local, anchor


def _mean(values, mask):
    return float(values[mask].mean()) if mask.any() else None


def pooled_local_similarity(local_tokens):
    """Mean-pool raw backbone patch tokens, then normalize each local vector."""
    if local_tokens.ndim != 3 or local_tokens.shape[0] != 8 or local_tokens.shape[1] == 0:
        raise ValueError("Expected eight nonempty local patch sequences [8, N, D]")
    pooled = local_tokens.float().mean(dim=1)
    if not torch.isfinite(pooled).all() or (pooled.norm(dim=-1) <= 1e-12).any():
        raise ValueError("Pooled local vectors must be finite and nonzero for cosine similarity")
    normalized = F.normalize(pooled, dim=-1)
    similarity = (normalized @ normalized.T).clamp(-1, 1)
    # Remove round-off on the mathematically unit self-cosines only after
    # rejecting undefined zero-vector cosines.
    similarity.fill_diagonal_(1.0)
    return similarity


def _warp(features, target_transforms, query_transforms, valid_masks, grid, patch_size):
    return warp_anchor_features_to_local(
        features, target_transforms, query_transforms,
        torch.zeros(len(query_transforms), dtype=torch.long), valid_masks,
        patch_size=patch_size, anchor_grid_size=grid[0], local_grid_size=grid[1],
    )


def analyze(local_tokens, anchor_tokens, views, *, negative_distance=0.1):
    """Use continuous geometry, never matching patch row/column numbers across crops.

    Local-local alignment is directed: query patch centres in crop i sample raw
    contextual tokens in crop j bilinearly; those tokens are then L2 normalized.
    Far negatives must exceed negative_distance * source-image diagonal.
    Positive and negative means use the same eligible query patches.
    """
    count, patches, _ = local_tokens.shape
    if count != 8 or patches != views.grid ** 2:
        raise ValueError("Local feature shapes do not match the eight input grids")
    local = F.normalize(local_tokens.float(), dim=-1)
    pooled_similarity = pooled_local_similarity(local_tokens)
    anchor = F.normalize(anchor_tokens.float(), dim=-1)
    flat = local.flatten(0, 1)
    similarity = (flat @ flat.T).clamp(-1, 1)
    blocks = similarity.reshape(count, patches, count, patches).permute(0, 2, 1, 3)
    pair_mean = blocks.mean((-1, -2))
    for i in range(count):
        pair_mean[i, i] = blocks[i, i][~torch.eye(patches, dtype=torch.bool)].mean()

    coords = views.source_centres
    far_cutoff = negative_distance * math.hypot(*views.source.size)
    aligned_cosine = torch.full((count, count, patches), float("nan"))
    far_cosine = torch.full_like(aligned_cosine, float("nan"))
    positive_valid = torch.zeros_like(aligned_cosine, dtype=torch.bool)
    comparison_valid = torch.zeros_like(positive_valid)
    all_valid = torch.ones(1, views.grid, views.grid, dtype=torch.bool)
    for j in range(count):
        warped, valid = _warp(local_tokens[j:j + 1], views.transforms[j:j + 1],
                              views.transforms, all_valid,
                              ((views.grid, views.grid), (views.grid, views.grid)), views.patch_size)
        matched = (local * F.normalize(warped, dim=-1)).sum(-1).clamp(-1, 1)
        valid[j] = False  # No self-view matches in correspondence metrics.
        aligned_cosine[:, j] = matched.masked_fill(~valid, float("nan"))
        positive_valid[:, j] = valid
        # Restrict negatives to the same query rows that have a valid match.
        far = torch.cdist(coords.flatten(0, 1), coords[j]).reshape(count, patches, patches) > far_cutoff
        negative_count = far.sum(-1)
        negative_mean = (blocks[:, j] * far).sum(-1) / negative_count.clamp_min(1)
        eligible = valid & (negative_count > 0)
        far_cosine[:, j] = negative_mean.masked_fill(~eligible, float("nan"))
        comparison_valid[:, j] = eligible

    anchor_grid = views.anchor.shape[-1] // views.patch_size
    warped_anchor, anchor_valid = _warp(
        anchor_tokens, views.anchor_geometry["transform"][None], views.transforms,
        views.anchor_geometry["valid_mask"][None],
        ((anchor_grid, anchor_grid), (views.grid, views.grid)), views.patch_size,
    )
    anchor_cosine = (local * F.normalize(warped_anchor, dim=-1)).sum(-1).clamp(-1, 1)
    anchor_cosine = anchor_cosine.masked_fill(~anchor_valid, float("nan"))

    # Feature retrieval is restricted to valid anchor tokens; padding cannot win.
    anchor_similarity = local @ anchor[0].T
    anchor_similarity[..., ~views.anchor_geometry["valid_mask"].flatten()] = -torch.inf
    retrieval_index = anchor_similarity.argmax(-1)
    anchor_centres = patch_centres(anchor_grid, views.patch_size)
    expected_xy = torch.stack([transform_points(views.anchor_geometry["transform"], c) for c in coords])
    retrieved_xy = anchor_centres[retrieval_index]
    retrieval_error = (retrieved_xy - expected_xy).norm(dim=-1) / views.patch_size
    retrieval_error = retrieval_error.masked_fill(~anchor_valid, float("nan"))

    inter_local = ~torch.eye(count, dtype=torch.bool)
    metrics = {
        "pooled_cross_local_mean_cosine": float(pooled_similarity[inter_local].mean()),
        "cross_local_mean_cosine": float(pair_mean[inter_local].mean()),
        "within_local_mean_cosine_excluding_self": float(pair_mean.diag().mean()),
        "local_local_aligned_cosine": _mean(aligned_cosine, positive_valid),
        "local_local_aligned_query_count": int(positive_valid.sum()),
        "paired_positive_cosine": _mean(aligned_cosine, comparison_valid),
        "paired_far_negative_cosine": _mean(far_cosine, comparison_valid),
        "paired_discrimination_gap": _mean(aligned_cosine - far_cosine, comparison_valid),
        "paired_query_count": int(comparison_valid.sum()),
        "local_anchor_aligned_cosine": _mean(anchor_cosine, anchor_valid),
        "local_anchor_valid_query_count": int(anchor_valid.sum()),
        "anchor_retrieval_mean_error_patches": _mean(retrieval_error, anchor_valid),
        "anchor_retrieval_within_1_patch": _mean((retrieval_error <= 1).float(), anchor_valid),
    }
    arrays = {
        "local_pooled_similarity": pooled_similarity.numpy(),
        "local_mean_patch_features": local_tokens.float().mean(dim=1).numpy(),
        "patch_similarity": similarity.numpy(), "local_pair_mean": pair_mean.numpy(),
        "local_local_aligned_cosine": aligned_cosine.numpy(),
        "local_local_far_negative_cosine": far_cosine.numpy(),
        "local_local_valid": positive_valid.numpy(), "paired_query_valid": comparison_valid.numpy(),
        "local_anchor_aligned_cosine": anchor_cosine.numpy(), "local_anchor_valid": anchor_valid.numpy(),
        "anchor_retrieval_index": retrieval_index.numpy(), "anchor_retrieval_error_patches": retrieval_error.numpy(),
        "anchor_expected_xy": expected_xy.numpy(), "anchor_retrieved_xy": retrieved_xy.numpy(),
    }
    return metrics, arrays


def metric_differences(before, after):
    return {key: after[key] - value if value is not None and after[key] is not None else None
            for key, value in before.items() if not key.endswith("count")}


def aggregate(records):
    """Report equal-weight image/seed means; patch pairs are not independent samples."""
    output = {}
    for section in ("before", "after", "difference_after_minus_before"):
        output[section] = {}
        for key in records[0][section]:
            if key.endswith("count"):
                continue
            values = [record[section][key] for record in records if record[section][key] is not None]
            output[section][key] = {"mean": float(np.mean(values)) if values else None,
                                    "sample_count": len(values)}
    return output

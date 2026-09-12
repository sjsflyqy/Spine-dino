"""Geometry-exact sampling of full-FOV teacher tokens for local views."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def _patch_centres(
    height: int,
    width: int,
    patch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) * patch_size
    xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) * patch_size
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    ones = torch.ones_like(grid_x)
    return torch.stack((grid_x, grid_y, ones), dim=-1)


def warp_anchor_features_to_local(
    anchor_features: torch.Tensor,
    anchor_transforms: torch.Tensor,
    local_transforms: torch.Tensor,
    local_sample_ids: torch.Tensor,
    anchor_valid_mask: torch.Tensor,
    *,
    patch_size: int,
    anchor_grid_size: Tuple[int, int],
    local_grid_size: Tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Warp contextual anchor tokens onto each local patch grid.

    Args:
        anchor_features: ``[B, Ha*Wa, D]`` teacher patch tokens.
        anchor_transforms: ``[B, 3, 3]`` source-to-anchor matrices.
        local_transforms: ``[L, 3, 3]`` source-to-local matrices.
        local_sample_ids: ``[L]`` source sample index for every local view.
        anchor_valid_mask: ``[B, Ha, Wa]`` padding-aware token validity.

    Returns:
        Aligned teacher tokens ``[L, Hl*Wl, D]`` and a boolean valid mask
        ``[L, Hl*Wl]``.  Sampling is performed in float32 because the target
        branch does not require gradients and matrix inversion in fp16 is
        unnecessarily fragile.
    """
    if anchor_features.ndim != 3:
        raise ValueError("anchor_features must have shape [B, N, D]")
    anchor_height, anchor_width = anchor_grid_size
    local_height, local_width = local_grid_size
    batch_size, token_count, feature_dim = anchor_features.shape
    if token_count != anchor_height * anchor_width:
        raise ValueError("anchor token count does not match anchor_grid_size")
    if anchor_valid_mask.shape != (batch_size, anchor_height, anchor_width):
        raise ValueError("anchor_valid_mask has an incompatible shape")
    if local_transforms.shape[0] != local_sample_ids.shape[0]:
        raise ValueError("local transform and sample-id counts differ")

    device = anchor_features.device
    sample_ids = local_sample_ids.to(device=device, dtype=torch.long)
    if sample_ids.numel() and (sample_ids.min() < 0 or sample_ids.max() >= batch_size):
        raise ValueError("local_sample_ids contains an out-of-range sample")

    anchor_t = anchor_transforms.to(device=device, dtype=torch.float32)[sample_ids]
    local_t = local_transforms.to(device=device, dtype=torch.float32)
    local_to_anchor = anchor_t @ torch.linalg.inv(local_t)

    centres = _patch_centres(
        local_height,
        local_width,
        patch_size,
        device=device,
        dtype=torch.float32,
    )
    centres = centres.reshape(1, -1, 3).expand(sample_ids.shape[0], -1, -1)
    anchor_points = torch.bmm(centres, local_to_anchor.transpose(1, 2))
    anchor_xy = anchor_points[..., :2] / anchor_points[..., 2:].clamp_min(1e-12)

    anchor_pixel_width = anchor_width * patch_size
    anchor_pixel_height = anchor_height * patch_size
    grid_x = 2.0 * anchor_xy[..., 0] / float(anchor_pixel_width) - 1.0
    grid_y = 2.0 * anchor_xy[..., 1] / float(anchor_pixel_height) - 1.0
    sampling_grid = torch.stack((grid_x, grid_y), dim=-1).reshape(
        -1, local_height, local_width, 2
    )

    anchor_maps = anchor_features.float().reshape(
        batch_size, anchor_height, anchor_width, feature_dim
    ).permute(0, 3, 1, 2)
    selected_maps = anchor_maps[sample_ids]
    aligned = F.grid_sample(
        selected_maps,
        sampling_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    aligned = aligned.permute(0, 2, 3, 1).reshape(-1, local_height * local_width, feature_dim)

    selected_valid = anchor_valid_mask.to(device=device, dtype=torch.float32)[sample_ids, None]
    valid_fraction = F.grid_sample(
        selected_valid,
        sampling_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).reshape(-1, local_height * local_width)
    in_bounds = (
        (anchor_xy[..., 0] >= 0.0)
        & (anchor_xy[..., 0] <= anchor_pixel_width)
        & (anchor_xy[..., 1] >= 0.0)
        & (anchor_xy[..., 1] <= anchor_pixel_height)
    )
    valid = (valid_fraction >= 1.0 - 1e-5) & in_bounds
    return aligned, valid


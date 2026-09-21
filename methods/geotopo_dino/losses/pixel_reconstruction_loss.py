"""Pixel targets share exactly the teacher global-view geometry and ordering."""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    if images.ndim != 4 or patch_size <= 0:
        raise ValueError("expected BCHW images and a positive patch_size")
    batch, channels, height, width = images.shape
    if height % patch_size or width % patch_size:
        raise ValueError("image dimensions must be divisible by patch_size")
    h, w = height // patch_size, width // patch_size
    return images.reshape(batch, channels, h, patch_size, w, patch_size).permute(
        0, 2, 4, 3, 5, 1
    ).reshape(batch, h * w, patch_size * patch_size * channels)


def unpatchify(patches: torch.Tensor, patch_size: int, grid_size: tuple[int, int]) -> torch.Tensor:
    batch, length, features = patches.shape
    h, w = grid_size
    if length != h * w or features % (patch_size * patch_size):
        raise ValueError("patch tensor does not match the supplied grid/patch size")
    channels = features // (patch_size * patch_size)
    return patches.reshape(batch, h, w, patch_size, patch_size, channels).permute(
        0, 5, 1, 3, 2, 4
    ).reshape(batch, channels, h * patch_size, w * patch_size)


def global_reconstruction_mask(masks: torch.Tensor, anchor_valid_masks: torch.Tensor) -> torch.Tensor:
    """Views are [all anchors, all random globals]; only anchors have padding."""
    valid = anchor_valid_masks.flatten(1).bool()
    if masks.ndim != 2 or masks.shape != (2 * valid.shape[0], valid.shape[1]):
        raise ValueError("expected two view-major global crops matching anchor validity")
    return masks.bool() & torch.cat((valid, torch.ones_like(valid)), dim=0)


def pixel_warmup_scale(iteration: int, warmup_iterations: int) -> float:
    if warmup_iterations < 0:
        raise ValueError("pixel warmup_iterations must be non-negative")
    return min(max(iteration, 0) / warmup_iterations, 1.0) if warmup_iterations else 1.0


class PixelReconstructionLoss(nn.Module):
    def __init__(self, patch_size: int, norm_pix_loss: bool = False):
        super().__init__()
        self.patch_size = patch_size
        self.norm_pix_loss = norm_pix_loss

    def forward(self, prediction: torch.Tensor, images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        # FP32 targets/MSE even when the backbone and decoder use mixed precision.
        target = patchify(images.detach().float(), self.patch_size)
        if prediction.shape != target.shape or masks.shape != target.shape[:2]:
            raise ValueError("pixel predictions, targets, and masks must share the same patch grid")
        masks = masks.bool()
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            # Matches MAE's sample variance (with a defined single-value case).
            variance = target.var(dim=-1, keepdim=True, unbiased=target.shape[-1] > 1)
            target = (target - mean) / (variance + 1e-6).sqrt()
        error = (prediction.float() - target).square().mean(dim=-1)
        numerator = error.masked_fill(~masks, 0.0).sum()
        count = masks.sum().to(dtype=torch.float32)
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            # FSDP averages gradients. This scaling gives the global masked-patch
            # mean even with unequal counts or a rank with no masked patches.
            world_size = dist.get_world_size()
            dist.all_reduce(count)
        # Keep the decoder/backbone in the graph even if every rank has zero masks.
        return numerator * world_size / count.clamp_min(1.0)

"""One-level Haar detail supervision on the existing masked RGB predictions."""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn

from .pixel_reconstruction_loss import patchify


def haar_details(patches: torch.Tensor) -> torch.Tensor:
    """BCHW -> B,3,C,H/2,W/2 orthonormal details; LL is deliberately omitted.

    The first two bands measure changes across rows and columns, respectively;
    the third measures diagonal changes. All arithmetic is FP32, including AMP.
    """
    if patches.ndim != 4 or min(patches.shape[-2:]) <= 0 or any(
        size % 2 for size in patches.shape[-2:]
    ):
        raise ValueError("Haar details require BCHW patches with positive even spatial sizes")
    patches = patches.float()
    a = patches[..., 0::2, 0::2]
    b = patches[..., 0::2, 1::2]
    c = patches[..., 1::2, 0::2]
    d = patches[..., 1::2, 1::2]
    return torch.stack(((a + b - c - d) / 2,
                        (a - b + c - d) / 2,
                        (a - b - c + d) / 2), dim=1)


class WaveletReconstructionLoss(nn.Module):
    """Equal-band high-frequency L1, averaged over globally masked patches.

    Uses the same normalized global crops and channels-last flattened patch
    layout as PixelReconstructionLoss. This module has no parameters or buffers.
    """

    def __init__(self, patch_size: int):
        super().__init__()
        if patch_size <= 0 or patch_size % 2:
            raise ValueError("wavelet reconstruction requires a positive even patch_size")
        self.patch_size = patch_size

    def forward(self, prediction: torch.Tensor, images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        target = patchify(images.detach().float(), self.patch_size)
        if prediction.shape != target.shape or masks.shape != target.shape[:2]:
            raise ValueError("wavelet predictions, targets, and masks must share the same patch grid")
        masks = masks.bool()
        # Select before transforming, but never skip a distributed collective on
        # an empty local mask. Empty selections also retain the prediction graph.
        channels = images.shape[1]
        p = self.patch_size
        predicted_patches = prediction[masks].float().reshape(-1, p, p, channels).permute(0, 3, 1, 2)
        target_patches = target[masks].reshape(-1, p, p, channels).permute(0, 3, 1, 2)
        error = (haar_details(predicted_patches) - haar_details(target_patches)).abs().mean(dim=(1, 2, 3, 4))
        numerator = error.sum()
        count = masks.sum().to(dtype=torch.float32)
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            # FSDP averages gradients: match the global masked-patch mean even
            # when ranks have different counts, including an empty local rank.
            world_size = dist.get_world_size()
            dist.all_reduce(count)
        return numerator * world_size / count.clamp_min(1.0)

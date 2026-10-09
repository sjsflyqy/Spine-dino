"""Sobel gradient supervision as an independent control for Haar detail loss."""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from .pixel_reconstruction_loss import patchify


def sobel_gradients(patches: torch.Tensor) -> torch.Tensor:
    """BCHW -> B,2,C,H-2,W-2 signed x/y gradients, with Sobel kernels /8.

    Valid convolution keeps every stencil inside the selected patch, without
    padding or borrowing visible/invalid neighboring patches. Convolution must
    explicitly leave autocast to retain FP32 under the training AMP context.
    """
    if patches.ndim != 4 or min(patches.shape[-2:]) < 3:
        raise ValueError("Sobel gradients require BCHW patches with spatial sizes at least 3")
    batch, channels, height, width = patches.shape
    with torch.autocast(device_type=patches.device.type, enabled=False):
        kernels = torch.tensor(
            [[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
             [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]],
            device=patches.device, dtype=torch.float32,
        ).unsqueeze(1) / 8
        gradients = F.conv2d(patches.float().reshape(-1, 1, height, width), kernels)
    return gradients.reshape(batch, channels, 2, height - 2, width - 2).permute(0, 2, 1, 3, 4)


class GradientReconstructionLoss(nn.Module):
    """Equal-direction Sobel L1, averaged over globally masked RGB patches.

    Reuses the pixel target space and channels-last flattened patch layout.
    This loss has no trainable parameters or checkpoint buffers.
    """

    def __init__(self, patch_size: int):
        super().__init__()
        if patch_size < 3:
            raise ValueError("gradient reconstruction requires patch_size at least 3")
        self.patch_size = patch_size

    def forward(self, prediction: torch.Tensor, images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        target = patchify(images.detach().float(), self.patch_size)
        if prediction.shape != target.shape or masks.shape != target.shape[:2]:
            raise ValueError("gradient predictions, targets, and masks must share the same patch grid")
        masks = masks.bool()
        p, channels = self.patch_size, images.shape[1]
        predicted_patches = prediction[masks].float().reshape(-1, p, p, channels).permute(0, 3, 1, 2)
        target_patches = target[masks].reshape(-1, p, p, channels).permute(0, 3, 1, 2)
        error = (sobel_gradients(predicted_patches) - sobel_gradients(target_patches)).abs().mean(dim=(1, 2, 3, 4))
        numerator = error.sum()
        count = masks.sum().to(dtype=torch.float32)
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            # Match FSDP gradient averaging, including ranks with zero masks.
            world_size = dist.get_world_size()
            dist.all_reduce(count)
        return numerator * world_size / count.clamp_min(1.0)

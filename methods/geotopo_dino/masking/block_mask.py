"""Current DINOv2 block-mask policy."""

from __future__ import annotations

import torch

from .base import MaskPolicy


class BlockMaskPolicy(MaskPolicy):
    """Select collate-generated valid-aware DINOv2 block masks unchanged."""

    def select(
        self,
        candidate_masks: torch.Tensor,
        *,
        teacher_anchor_tokens: torch.Tensor,
        anchor_valid_mask: torch.Tensor,
        progress: float,
    ) -> tuple[torch.Tensor, None]:
        del teacher_anchor_tokens, anchor_valid_mask, progress
        return candidate_masks, None


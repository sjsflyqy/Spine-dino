"""Current DINOv2 block-mask policy."""

from __future__ import annotations

import torch

from .base import MaskPolicy


class BlockMaskPolicy(MaskPolicy):
    """Select collate-generated valid-aware DINOv2 block masks unchanged."""

    @property
    def signature(self):
        return {"anchor_policy": "block", "random_global_policy": "block"}

    def select(
        self,
        candidate_masks: torch.Tensor,
        *,
        teacher_anchor_tokens: torch.Tensor,
        anchor_valid_mask: torch.Tensor,
        progress: float,
        iteration: int | None = None,
    ) -> tuple[torch.Tensor, None]:
        del teacher_anchor_tokens, anchor_valid_mask, progress, iteration
        return candidate_masks, None

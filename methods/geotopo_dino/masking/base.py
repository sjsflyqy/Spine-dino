"""Mask-policy interface shared by block masking and future TGSR masking."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class MaskPolicy(ABC):
    @abstractmethod
    def select(
        self,
        candidate_masks: torch.Tensor,
        *,
        teacher_anchor_tokens: torch.Tensor,
        anchor_valid_mask: torch.Tensor,
        progress: float,
    ) -> tuple[torch.Tensor, object | None]:
        """Return final global masks and optional topology state."""
        raise NotImplementedError


"""Shared output contract for the future teacher-guided topology policy."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TopologyState:
    centerline: torch.Tensor
    part_masks: torch.Tensor
    part_valid: torch.Tensor
    masked_parts: torch.Tensor
    confidence: torch.Tensor
    used_fallback: torch.Tensor


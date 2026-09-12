"""Convert a final mask policy output to the representation used by iBOT."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MaskState:
    masks: torch.Tensor
    indices: torch.Tensor
    weights: torch.Tensor
    n_masked_patches_tensor: torch.Tensor
    upperbound: int
    topology: object | None = None

    @property
    def n_masked_patches(self) -> int:
        return int(self.indices.shape[0])


def pack_masks(
    masks: torch.Tensor,
    *,
    upperbound: int,
    topology: object | None = None,
) -> MaskState:
    masks = masks.bool()
    indices = masks.flatten().nonzero().flatten()
    weights = (1.0 / masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(masks)[masks]
    n_masked = torch.full(
        (1,),
        fill_value=indices.shape[0],
        dtype=torch.long,
        device=masks.device,
    )
    if upperbound < indices.shape[0]:
        raise ValueError("mask upperbound is smaller than the selected patch count")
    return MaskState(
        masks=masks,
        indices=indices,
        weights=weights,
        n_masked_patches_tensor=n_masked,
        upperbound=int(upperbound),
        topology=topology,
    )


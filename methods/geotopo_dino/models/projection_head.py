"""Small EMA teacher/student projection head used by both GCVD losses."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from dinov2.layers import DINOHead


class GeometryProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 256, hidden_dim: int = 2048) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens):
        return F.normalize(self.mlp(tokens), dim=-1, eps=1e-6)


class GCVDPrototypeHead(DINOHead):
    """Independent DINO-style projector/prototype head for dense GCVD."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        nlayers: int = 3,
    ) -> None:
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            nlayers=nlayers,
        )

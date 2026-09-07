"""Small dependency-free LoRA implementation for the downstream ViT backbones."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import nn


class LoRALinear(nn.Module):
    """A frozen Linear layer with a trainable low-rank residual branch."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.base = base
        self.base.requires_grad_(False)
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scaling = float(alpha) / rank
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base(inputs) + self.lora_B(self.lora_A(self.dropout(inputs))) * self.scaling


class LoRAQKVLinear(nn.Module):
    """Three independent rank-r adapters for a fused ViT QKV projection."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if base.out_features != 3 * base.in_features:
            raise ValueError(
                f"Fused qkv must have out_features=3*in_features, got "
                f"{base.in_features}->{base.out_features}"
            )
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.base = base
        self.base.requires_grad_(False)
        self.lora_A = nn.ModuleList([
            nn.Linear(base.in_features, rank, bias=False) for _ in range(3)
        ])
        self.lora_B = nn.ModuleList([
            nn.Linear(rank, base.in_features, bias=False) for _ in range(3)
        ])
        self.dropout = nn.Dropout(dropout)
        self.scaling = float(alpha) / rank
        for projection_a, projection_b in zip(self.lora_A, self.lora_B):
            nn.init.kaiming_uniform_(projection_a.weight, a=math.sqrt(5))
            nn.init.zeros_(projection_b.weight)

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        dropped = self.dropout(inputs)
        residual = torch.cat([
            projection_b(projection_a(dropped))
            for projection_a, projection_b in zip(self.lora_A, self.lora_B)
        ], dim=-1)
        return self.base(inputs) + residual * self.scaling


def inject_lora(
    model: nn.Module,
    *,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
    target_modules: Iterable[str] = ("qkv", "query", "key", "value"),
) -> list[str]:
    """Freeze ``model`` and replace matching attention projections with LoRA."""

    model.requires_grad_(False)
    targets = set(target_modules)
    replacements: list[tuple[str, nn.Module, str, nn.Linear]] = []
    for full_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf = full_name.rsplit(".", 1)[-1]
        if leaf not in targets:
            continue
        parent_name, _, child_name = full_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        replacements.append((full_name, parent, child_name, module))

    if not replacements:
        raise RuntimeError(
            f"No Linear modules matched LoRA targets {sorted(targets)}; "
            "inspect the backbone's attention projection names"
        )
    for full_name, parent, child_name, module in replacements:
        wrapper = LoRAQKVLinear if full_name.rsplit(".", 1)[-1] == "qkv" else LoRALinear
        setattr(parent, child_name, wrapper(module, rank, alpha, dropout))
    return [name for name, _, _, _ in replacements]

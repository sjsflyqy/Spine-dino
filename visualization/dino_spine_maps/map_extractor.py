"""Extract layer-wise CLS attention and CLS-to-patch cosine maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from .model_loader import LoadedDino


@dataclass(frozen=True)
class LayerMaps:
    layer: int  # One-based layer number for user-facing output.
    attention_heads: np.ndarray  # [heads, grid_h, grid_w]
    attention_mean: np.ndarray  # [grid_h, grid_w]
    attention_max: np.ndarray  # [grid_h, grid_w]
    attention_patch_mass: np.ndarray  # [heads]
    cosine: np.ndarray  # [grid_h, grid_w]


@dataclass(frozen=True)
class ExtractionResult:
    layers: tuple[LayerMaps, ...]
    grid_size: tuple[int, int]
    final_cls_token: torch.Tensor
    final_patch_tokens: torch.Tensor


def _normalize_requested_layers(layers: Iterable[int] | None, depth: int) -> set[int]:
    if layers is None:
        return set(range(depth))
    selected = set(layers)
    if not selected:
        raise ValueError("At least one layer must be selected")
    invalid = sorted(index + 1 for index in selected if index < 0 or index >= depth)
    if invalid:
        raise ValueError(f"Layer numbers outside 1..{depth}: {invalid}")
    return selected


def _cls_attention(
    attention_module,
    normalized_tokens: torch.Tensor,
    *,
    patch_start: int,
    grid_height: int,
    grid_width: int,
    rope=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    batch, token_count, channels = normalized_tokens.shape
    if batch != 1:
        raise ValueError("Visualization extraction currently expects a single image")
    heads = int(attention_module.num_heads)
    head_dim = channels // heads
    qkv = attention_module.qkv(normalized_tokens).reshape(
        batch, token_count, 3, heads, head_dim
    )
    query, key, _ = torch.unbind(qkv, dim=2)
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    if rope is not None:
        query, key = attention_module.apply_rope(query, key, rope)

    # Only the CLS query row is required for a spatial attention map.
    logits = torch.matmul(
        query[:, :, :1].float(), key.float().transpose(-2, -1)
    ) * float(attention_module.scale)
    full_weights = torch.softmax(logits, dim=-1)[0, :, 0]
    patch_weights = full_weights[:, patch_start:]
    expected = grid_height * grid_width
    if patch_weights.shape[-1] != expected:
        raise RuntimeError(
            f"Attention has {patch_weights.shape[-1]} patch keys, expected {expected}"
        )
    patch_mass = patch_weights.sum(dim=-1)
    heads_grid = patch_weights.reshape(heads, grid_height, grid_width)
    mean_grid = heads_grid.mean(dim=0)
    max_grid = heads_grid.max(dim=0).values
    return tuple(
        value.detach().cpu().numpy().astype(np.float32, copy=False)
        for value in (heads_grid, mean_grid, max_grid, patch_mass)
    )


def _normalized_layer_tokens(loaded: LoadedDino, tokens: torch.Tensor, patch_start: int):
    model = loaded.model
    if loaded.generation == 3 and bool(getattr(model, "untie_cls_and_patch_norms", False)):
        cls_and_extra = model.cls_norm(tokens[:, :patch_start])
        patches = model.norm(tokens[:, patch_start:])
        return cls_and_extra[:, 0], patches
    normalized = model.norm(tokens)
    return normalized[:, 0], normalized[:, patch_start:]


def _cosine_grid(
    cls_token: torch.Tensor,
    patch_tokens: torch.Tensor,
    grid_height: int,
    grid_width: int,
) -> np.ndarray:
    cls_float = F.normalize(cls_token.float(), dim=-1)
    patch_float = F.normalize(patch_tokens.float(), dim=-1)
    values = (patch_float * cls_float.unsqueeze(1)).sum(dim=-1)
    return (
        values[0]
        .reshape(grid_height, grid_width)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )


@torch.inference_mode()
def extract_layer_maps(
    loaded: LoadedDino,
    image: torch.Tensor,
    *,
    layers: Iterable[int] | None = None,
) -> ExtractionResult:
    """Run one image through every block and retain maps for selected layers.

    ``layers`` uses zero-based indices internally; the CLI accepts one-based
    layer numbers and converts them before calling this function.
    """

    model = loaded.model
    selected = _normalize_requested_layers(layers, loaded.num_layers)
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError(f"Expected image shape [3,H,W] or [1,3,H,W], got {tuple(image.shape)}")
    grid_height = image.shape[-2] // loaded.patch_size
    grid_width = image.shape[-1] // loaded.patch_size
    patch_start = 1 + loaded.num_extra_tokens

    if loaded.generation == 2:
        tokens = model.prepare_tokens_with_masks(image, masks=None)
        rope = None
    else:
        tokens, prepared_grid = model.prepare_tokens_with_masks(image, masks=None)
        if tuple(prepared_grid) != (grid_height, grid_width):
            raise RuntimeError(
                f"DINOv3 prepared grid {prepared_grid} != expected {(grid_height, grid_width)}"
            )
        rope = (
            model.rope_embed(H=grid_height, W=grid_width)
            if model.rope_embed is not None
            else None
        )

    extracted: list[LayerMaps] = []
    final_cls = None
    final_patches = None
    for index, block in enumerate(model.blocks):
        normalized_input = block.norm1(tokens)
        if index in selected:
            heads, mean, maximum, patch_mass = _cls_attention(
                block.attn,
                normalized_input,
                patch_start=patch_start,
                grid_height=grid_height,
                grid_width=grid_width,
                rope=rope,
            )

        tokens = block(tokens) if loaded.generation == 2 else block(tokens, rope)
        cls_token, patch_tokens = _normalized_layer_tokens(loaded, tokens, patch_start)
        final_cls, final_patches = cls_token, patch_tokens
        if index in selected:
            extracted.append(
                LayerMaps(
                    layer=index + 1,
                    attention_heads=heads,
                    attention_mean=mean,
                    attention_max=maximum,
                    attention_patch_mass=patch_mass,
                    cosine=_cosine_grid(
                        cls_token, patch_tokens, grid_height, grid_width
                    ),
                )
            )

    assert final_cls is not None and final_patches is not None
    return ExtractionResult(
        layers=tuple(extracted),
        grid_size=(grid_height, grid_width),
        final_cls_token=final_cls.detach().cpu(),
        final_patch_tokens=final_patches.detach().cpu(),
    )


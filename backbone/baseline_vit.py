"""Adapters for the public DAX, Medical-MAE and CheXFound encoders.

The three projects use closely related ViT implementations, but their
checkpoints are wrapped differently.  This module keeps that checkpoint
handling at the boundary and exposes the same :class:`BackboneOutput` used by
the existing DINO downstream tasks.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re
from typing import Any

import torch
from torch import Tensor, nn

from baseline_pretrain.models import VisionTransformer

from .checkpoints import load_checkpoint
from .registry import BackboneOutput


BASELINE_SPECS: dict[str, dict[str, Any]] = {
    "dax": dict(dim=384, depth=12, heads=6, image_size=224, patch_size=16,
                registers=0, swiglu=False, layerscale=False, drop_path=0.0,
                drop_path_uniform=False, antialias=True, pos_offset=0.1),
    "medical_mae": dict(dim=768, depth=12, heads=12, image_size=224, patch_size=16,
                        registers=0, swiglu=False, layerscale=False, drop_path=0.0,
                        drop_path_uniform=False, antialias=False, pos_offset=0.0),
    "chexfound": dict(dim=1024, depth=24, heads=16, image_size=512, patch_size=16,
                      registers=4, swiglu=True, layerscale=True, drop_path=0.0,
                      drop_path_uniform=True, antialias=False, pos_offset=0.1),
}


def _tensor_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and bool(value) and all(
        isinstance(key, str) and isinstance(item, torch.Tensor)
        for key, item in value.items()
    )


def _find_state(value: Any) -> dict[str, Tensor] | None:
    """Find the first tensor mapping in a known SSL-checkpoint wrapper."""
    if _tensor_mapping(value):
        return dict(value)
    if not isinstance(value, Mapping):
        return None
    # Prefer the encoder-bearing branches.  A CheXFound full checkpoint may
    # contain both teacher and student; the teacher is the evaluated model.
    for key in ("teacher", "model", "state_dict", "backbone", "student"):
        if key in value:
            found = _find_state(value[key])
            if found is not None:
                return found
    for item in value.values():
        found = _find_state(item)
        if found is not None:
            return found
    return None


def _normalize_keys(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    state = dict(state)
    # Repeated prefixes occur in DDP, FSDP and the CheXFound teacher export.
    prefixes = (
        "module.", "model.", "teacher.", "student.",
        "backbone.", "_fsdp_wrapped_module.", "_orig_mod.",
    )
    changed = True
    while changed:
        changed = False
        updated = {}
        for key, value in state.items():
            new_key = key
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
            updated[new_key] = value
        state = updated
    # CheXFound's official teacher was saved with four BlockChunk containers.
    # The second numeric component is the original global block index, so
    # blocks.2.12.attn.qkv becomes blocks.12.attn.qkv.
    chunked = {}
    for key, value in state.items():
        match = re.match(r"^blocks\.\d+\.(\d+)\.(.+)$", key)
        normalized_key = (
            "blocks." + match.group(1) + "." + match.group(2)
            if match else key
        )
        chunked[normalized_key] = value
    state = chunked
    return state


class BaselineVisionTransformer(VisionTransformer):
    """The shared ViT with the intermediate-layer method used by task 07."""

    def get_intermediate_layers(self, images: Tensor, n, reshape: bool = False):
        indices = tuple(int(index) for index in n)
        if not indices or min(indices) < 0 or max(indices) >= len(self.blocks):
            raise ValueError(f"Invalid intermediate block indices: {indices}")
        h, w = images.shape[-2:]
        if h % self.patch_size or w % self.patch_size:
            raise ValueError("Image height/width must be divisible by patch size")
        x = self.patch_embed(images)
        x = torch.cat((self.cls_token.expand(len(x), -1, -1), x), 1)
        x = x + self.position(h, w)
        if self.register_count:
            x = torch.cat((x[:, :1], self.register_tokens.expand(len(x), -1, -1), x[:, 1:]), 1)
        outputs = []
        wanted = set(indices)
        for index, block in enumerate(self.blocks):
            x = block(x)
            if index in wanted:
                normalized = self.norm(x)
                tokens = normalized[:, 1 + self.register_count:]
                if reshape:
                    gh, gw = h // self.patch_size, w // self.patch_size
                    outputs.append(tokens.transpose(1, 2).reshape(len(x), self.dim, gh, gw))
                else:
                    outputs.append(normalized)
        return tuple(outputs)


class BaselineViTBackbone(nn.Module):
    """DAX/Medical-MAE/CheXFound adapter with a common feature protocol."""

    def __init__(self, name: str, weights: str, *, freeze: bool = True):
        super().__init__()
        key = name.lower().replace("-", "_")
        if key not in BASELINE_SPECS:
            raise ValueError(f"Unknown baseline backbone: {name!r}")
        path = Path(weights).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Backbone weights not found: {path}")
        self.name = key
        self.weights = str(path)
        spec = dict(BASELINE_SPECS[key])
        self.model = BaselineVisionTransformer(**spec)
        self.patch_size = int(spec["patch_size"])
        self.embedding_dim = int(spec["dim"])
        self._load(path)
        if freeze:
            self.freeze()

    def _load(self, path: Path) -> None:
        # DAX's published file contains legacy NumPy metadata; this fallback
        # is restricted to the explicitly selected local baseline checkpoint.
        payload = load_checkpoint(path, allow_legacy_pickle=True)
        state = _find_state(payload)
        if state is None:
            raise ValueError(f"{path} does not contain a tensor state dictionary")
        state = _normalize_keys(state)
        expected = self.model.state_dict()
        selected: dict[str, Tensor] = {}
        shape_errors = []
        for key, value in state.items():
            if key not in expected:
                continue  # DINO/iBOT heads and MAE decoders are intentionally ignored.
            if tuple(value.shape) != tuple(expected[key].shape):
                shape_errors.append(f"{key}: checkpoint={tuple(value.shape)}, model={tuple(expected[key].shape)}")
            else:
                selected[key] = value
        if shape_errors:
            raise ValueError(f"{self.name} checkpoint shape mismatch in {path}: {shape_errors[:5]}")
        missing = sorted(set(expected) - set(selected))
        if missing:
            raise ValueError(
                f"{self.name} checkpoint is missing encoder parameters in {path}: {missing[:12]}"
            )
        self.model.load_state_dict(selected, strict=True)

    def freeze(self) -> None:
        self.model.requires_grad_(False)
        self.model.eval()

    def unfreeze(self) -> None:
        self.model.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        if not any(parameter.requires_grad for parameter in self.model.parameters()):
            self.model.eval()
        return self

    def forward(self, pixel_values: Tensor) -> BackboneOutput:
        cls_token, patch_tokens = self.model(pixel_values)
        grid_h = pixel_values.shape[-2] // self.patch_size
        grid_w = pixel_values.shape[-1] // self.patch_size
        expected = grid_h * grid_w
        if patch_tokens.shape[1] != expected:
            raise RuntimeError(
                f"Cannot reshape {patch_tokens.shape[1]} tokens to {grid_h}x{grid_w}"
            )
        feature_map = patch_tokens.transpose(1, 2).reshape(
            pixel_values.shape[0], self.embedding_dim, grid_h, grid_w
        )
        return BackboneOutput(
            cls_token=cls_token,
            patch_tokens=patch_tokens,
            feature_map=feature_map,
            patch_size=self.patch_size,
            embedding_dim=self.embedding_dim,
            grid_size=(grid_h, grid_w),
        )


"""Shared extraction and architecture checks for native DINO checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
import pickle

import torch


def load_checkpoint(path: Path, *, allow_legacy_pickle: bool = False) -> Any:
    """Read checkpoints, retaining compatibility with old public exports.

    The DAX public checkpoint contains a NumPy scalar in its metadata, which
    PyTorch 2.5 rejects under ``weights_only=True``.  Baseline adapters pass
    ``allow_legacy_pickle=True`` only for trusted local reference weights;
    native DINO loaders retain the strict safe-unpickling default.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except (RuntimeError, pickle.UnpicklingError) as exc:
        if isinstance(exc, pickle.UnpicklingError) and allow_legacy_pickle:
            return torch.load(path, map_location="cpu", weights_only=False)
        if "mmap can only be used with files saved with" not in str(exc):
            raise
        return torch.load(path, map_location="cpu", weights_only=True)


def _is_tensor_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and bool(value) and all(
        isinstance(key, str) and isinstance(tensor, torch.Tensor)
        for key, tensor in value.items()
    )


def extract_state_dict(checkpoint: Any, path: Path) -> dict[str, torch.Tensor]:
    """Accept bare backbones and known exports, preferring explicit teachers."""
    if _is_tensor_mapping(checkpoint):
        state = dict(checkpoint)
    elif isinstance(checkpoint, Mapping):
        state = None
        for key in ("teacher", "model", "state_dict", "backbone"):
            candidate = checkpoint.get(key)
            if _is_tensor_mapping(candidate):
                state = dict(candidate)
                break
        if state is None:
            available = ", ".join(map(str, list(checkpoint)[:12]))
            raise ValueError(
                f"{path} is not an extracted backbone checkpoint (top-level keys: {available}). "
                "Use tools/extract_teacher_backbone.py on a full SSL checkpoint first."
            )
    else:
        raise TypeError(f"Unsupported checkpoint object in {path}: {type(checkpoint).__name__}")

    # Strip uniform export/DDP prefixes, including repeated wrapper combinations.
    while True:
        prefix = next((prefix for prefix in ("module.", "model.")
                       if all(key.startswith(prefix) for key in state)), None)
        if prefix is None:
            break
        state = {key[len(prefix):]: value for key, value in state.items()}
    for prefix in (
        "teacher.backbone._fsdp_wrapped_module.", "teacher.backbone.",
        "backbone._fsdp_wrapped_module.", "backbone.",
    ):
        matching = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
        if matching:
            state = matching
            break
    return state


def validate_native_architecture(state: Mapping[str, torch.Tensor], generation: int, path: Path) -> None:
    """Reject conflicting architecture choices before constructing a large model."""
    detected = None
    if "storage_tokens" in state or any(key.startswith("rope_embed.") for key in state):
        detected = 3
    elif "pos_embed" in state:
        detected = 2
    projection = state.get("patch_embed.proj.weight")
    patch_shape = tuple(projection.shape[-2:]) if projection is not None and projection.ndim == 4 else None
    if detected is None:
        detected = {(14, 14): 2, (16, 16): 3}.get(patch_shape)
    if detected is not None and detected != generation:
        raise ValueError(
            f"Backbone architecture mismatch: config selects DINOv{generation}, "
            f"but checkpoint {path} contains DINOv{detected} weights. "
            f"Use a config with model.backbone: dinov{detected}, or matching weights. "
            "--weights overrides the file path only; it does not change model.backbone."
        )
    expected_patch = 14 if generation == 2 else 16
    if patch_shape is not None and patch_shape != (expected_patch, expected_patch):
        raise ValueError(
            f"DINOv{generation} ViT-B expects patch size {expected_patch}, "
            f"but checkpoint {path} has patch shape {patch_shape}. Use matching backbone weights."
        )

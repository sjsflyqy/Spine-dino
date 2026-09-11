"""Load repository DINOv2/DINOv3 backbone checkpoints for visualization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Any, Literal

import torch
from torch import nn


Generation = Literal[2, 3]


@dataclass(frozen=True)
class LoadedDino:
    """A loaded native DINO ViT plus the metadata needed by extractors."""

    model: nn.Module
    generation: Generation
    weight_path: Path
    patch_size: int
    embedding_dim: int
    num_layers: int
    num_heads: int
    num_extra_tokens: int

    @property
    def architecture(self) -> str:
        return "dinov2_vitb14" if self.generation == 2 else "dinov3_vitb16"


def _torch_load(path: Path) -> Any:
    kwargs = {"map_location": "cpu"}
    try:
        return torch.load(path, weights_only=True, mmap=True, **kwargs)
    except TypeError:
        try:
            return torch.load(path, weights_only=True, **kwargs)
        except TypeError:
            return torch.load(path, **kwargs)


def _is_tensor_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and bool(value) and all(
        isinstance(key, str) and isinstance(tensor, torch.Tensor)
        for key, tensor in value.items()
    )


def _strip_uniform_prefix(state: Mapping[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    if state and all(key.startswith(prefix) for key in state):
        return {key[len(prefix) :]: value for key, value in state.items()}
    return dict(state)


def _extract_state_dict(checkpoint: Any, path: Path) -> dict[str, torch.Tensor]:
    """Accept bare backbones and common wrappers, rejecting ambiguous full SSL states."""

    if _is_tensor_mapping(checkpoint):
        state = dict(checkpoint)
    elif isinstance(checkpoint, Mapping):
        state = None
        for key in ("model", "state_dict", "backbone"):
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

    # DataParallel and some exported wrappers add a uniform prefix.
    for prefix in ("module.", "model."):
        state = _strip_uniform_prefix(state, prefix)

    # A state dict may contain a complete teacher module. Keep only its backbone.
    for prefix in ("teacher.backbone.", "backbone."):
        matching = {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
        if matching:
            state = matching
            break
    return state


def _detect_generation(state: Mapping[str, torch.Tensor], requested: str) -> Generation:
    if requested == "dinov2":
        return 2
    if requested == "dinov3":
        return 3

    keys = set(state)
    if "pos_embed" in keys:
        return 2
    if "storage_tokens" in keys or "rope_embed.periods" in keys:
        return 3
    if any(key.startswith("rope_embed.") for key in keys):
        return 3
    raise ValueError(
        "Could not infer architecture from checkpoint keys. Pass --architecture dinov2 or dinov3."
    )


def _build_dinov2(repo_root: Path, *, device: torch.device) -> nn.Module:
    # The upstream module treats an installed xFormers package as available,
    # but its memory-efficient attention kernel cannot execute on CPU.
    if device.type == "cpu":
        os.environ["XFORMERS_DISABLED"] = "1"
    source_root = repo_root / "upstream" / "dinov2-main"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from dinov2.models.vision_transformer import vit_base

    return vit_base(patch_size=14, img_size=518, init_values=1.0, block_chunks=0)


def _build_dinov3(repo_root: Path) -> nn.Module:
    source_root = repo_root / "upstream" / "dinov3-main"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from dinov3.hub.backbones import dinov3_vitb16

    return dinov3_vitb16(pretrained=False)


def load_dino_model(
    weights: str | Path,
    *,
    architecture: str = "auto",
    device: str | torch.device = "cpu",
) -> LoadedDino:
    """Load a native ViT-B backbone from an extracted ``.pth`` file."""

    if architecture not in {"auto", "dinov2", "dinov3"}:
        raise ValueError(f"Unsupported architecture selection: {architecture}")
    path = Path(weights).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    if path.suffix.lower() not in {".pth", ".pt"}:
        raise ValueError(f"Expected a native .pth/.pt backbone checkpoint, got: {path}")

    checkpoint = _torch_load(path)
    state = _extract_state_dict(checkpoint, path)
    generation = _detect_generation(state, architecture)
    target_device = torch.device(device)
    repo_root = Path(__file__).resolve().parents[2]
    model = (
        _build_dinov2(repo_root, device=target_device)
        if generation == 2
        else _build_dinov3(repo_root)
    )

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint {path} was detected as DINOv{generation}, but it does not match the "
            f"repository ViT-B architecture. Original error:\n{exc}"
        ) from exc

    model.requires_grad_(False)
    model.eval()
    model.to(target_device)

    blocks = list(model.blocks)
    if not blocks:
        raise RuntimeError("Loaded model has no Transformer blocks")
    patch_size_value = getattr(model, "patch_size", 14 if generation == 2 else 16)
    if isinstance(patch_size_value, tuple):
        if patch_size_value[0] != patch_size_value[1]:
            raise ValueError(f"Non-square patch size is unsupported: {patch_size_value}")
        patch_size_value = patch_size_value[0]
    extra_tokens = (
        int(getattr(model, "num_register_tokens", 0))
        if generation == 2
        else int(getattr(model, "n_storage_tokens", 0))
    )
    return LoadedDino(
        model=model,
        generation=generation,
        weight_path=path,
        patch_size=int(patch_size_value),
        embedding_dim=int(getattr(model, "embed_dim", blocks[0].attn.qkv.in_features)),
        num_layers=len(blocks),
        num_heads=int(blocks[0].attn.num_heads),
        num_extra_tokens=extra_tokens,
    )

"""Shared visual-backbone interface for every downstream task."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Optional

import torch
from torch import Tensor, nn

from .lora import inject_lora


@dataclass
class BackboneOutput:
    cls_token: Tensor
    patch_tokens: Tensor
    feature_map: Tensor
    patch_size: int
    embedding_dim: int
    grid_size: tuple[int, int]


class HFVisualBackbone(nn.Module):
    """Adapter for Hugging Face ViT backbones with DINO-style outputs."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        freeze: bool = True,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required; install requirements-downstream.txt"
            ) from exc

        source = Path(model_name_or_path)
        raw_safetensors = source / "rad_dino_model.safetensors"
        if source.is_dir() and raw_safetensors.is_file() and not (source / "config.json").is_file():
            from safetensors.torch import load_file
            from transformers import Dinov2Config, Dinov2Model

            config = Dinov2Config(image_size=518, patch_size=14, hidden_size=768,
                                  num_hidden_layers=12, num_attention_heads=12)
            self.model = Dinov2Model(config)
            self.model.load_state_dict(load_file(str(raw_safetensors)), strict=True)
        else:
            self.model = AutoModel.from_pretrained(
                model_name_or_path,
                local_files_only=local_files_only,
            )
        self.model_name_or_path = model_name_or_path
        self.patch_size = int(getattr(self.model.config, "patch_size", 14))
        self.embedding_dim = int(
            getattr(self.model.config, "hidden_size", 768)
        )
        if freeze:
            self.freeze()

    def freeze(self) -> None:
        self.model.requires_grad_(False)
        self.model.eval()

    def unfreeze(self) -> None:
        self.model.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        if not any(p.requires_grad for p in self.model.parameters()):
            self.model.eval()
        return self

    def forward(self, pixel_values: Tensor) -> BackboneOutput:
        # The keypoint probes use rectangular radiographs. DINO-style HF
        # models need positional embeddings interpolated away from their
        # pretraining resolution; older compatible models may not expose the
        # keyword, hence the narrow fallback.
        try:
            outputs = self.model(
                pixel_values=pixel_values,
                return_dict=True,
                interpolate_pos_encoding=True,
            )
        except TypeError:
            outputs = self.model(pixel_values=pixel_values, return_dict=True)
        tokens = outputs.last_hidden_state
        if tokens.ndim != 3 or tokens.shape[1] < 2:
            raise RuntimeError(f"Unexpected backbone output: {tuple(tokens.shape)}")

        cls_token = tokens[:, 0]
        patch_tokens = tokens[:, 1:]
        input_h, input_w = pixel_values.shape[-2:]
        grid_h = input_h // self.patch_size
        grid_w = input_w // self.patch_size
        expected = grid_h * grid_w

        # DINO variants may add register tokens between CLS and patch tokens.
        if patch_tokens.shape[1] != expected:
            patch_tokens = patch_tokens[:, -expected:]
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


class NativeDinoBackbone(nn.Module):
    """Adapter for local official DINOv2/DINOv3 PyTorch checkpoints."""

    def __init__(self, generation: int, weights: str, *, freeze: bool = True) -> None:
        super().__init__()
        repo_root = Path(__file__).resolve().parents[1]
        weight_path = Path(weights).expanduser().resolve()
        if not weight_path.is_file():
            raise FileNotFoundError(f"Backbone weights not found: {weight_path}")

        if generation == 2:
            source_root = repo_root / "upstream" / "dinov2-main"
            sys.path.insert(0, str(source_root))
            from dinov2.models.vision_transformer import vit_base

            self.model = vit_base(
                patch_size=14, img_size=518, init_values=1.0, block_chunks=0
            )
            checkpoint = torch.load(
                weight_path, map_location="cpu", weights_only=True, mmap=True
            )
            # SSL initialization files retain architecture metadata and wrap
            # the actual DINOv2 backbone under ``model``.  Extracted downstream
            # teacher files are already bare state dictionaries.  Supporting
            # both formats keeps the downstream CLI compatible with every
            # checkpoint produced by this repository.
            state_dict = checkpoint.get("model", checkpoint)
            if not isinstance(state_dict, dict):
                raise TypeError(
                    f"Invalid DINOv2 checkpoint state in {weight_path}: "
                    f"{type(state_dict).__name__}"
                )
            self.model.load_state_dict(state_dict, strict=True)
            self.patch_size = 14
        elif generation == 3:
            source_root = repo_root / "upstream" / "dinov3-main"
            sys.path.insert(0, str(source_root))
            from dinov3.hub.backbones import dinov3_vitb16

            self.model = dinov3_vitb16(weights=str(weight_path))
            self.patch_size = 16
        else:
            raise ValueError(f"Unsupported DINO generation: {generation}")

        self.embedding_dim = 768
        if freeze:
            self.freeze()

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
        output = self.model.forward_features(pixel_values)
        cls_token = output["x_norm_clstoken"]
        patch_tokens = output["x_norm_patchtokens"]
        grid_h = pixel_values.shape[-2] // self.patch_size
        grid_w = pixel_values.shape[-1] // self.patch_size
        if patch_tokens.shape[1] != grid_h * grid_w:
            raise RuntimeError(
                f"Cannot reshape {patch_tokens.shape[1]} tokens to {grid_h}x{grid_w}"
            )
        feature_map = patch_tokens.transpose(1, 2).reshape(
            pixel_values.shape[0], self.embedding_dim, grid_h, grid_w
        )
        return BackboneOutput(cls_token, patch_tokens, feature_map, self.patch_size,
                              self.embedding_dim, (grid_h, grid_w))


def build_backbone(
    name: str,
    weights: Optional[str] = None,
    *,
    freeze: bool = True,
    local_files_only: bool = False,
    adapter: Optional[dict[str, Any]] = None,
) -> nn.Module:
    aliases = {
        "rad_dino": "microsoft/rad-dino",
        "rad_dino_maira2": "microsoft/rad-dino-maira-2",
        "rad_dino_maira_2": "microsoft/rad-dino-maira-2",
        "dinov2": "facebook/dinov2-base",
        "dinov2_base": "facebook/dinov2-base",
    }
    key = name.lower().replace("-", "_")
    if key in {"dinov2", "dinov2_base"} and weights and Path(weights).suffix == ".pth":
        backbone = NativeDinoBackbone(2, weights, freeze=freeze)
        return _configure_adapter(backbone, adapter)
    if key in {"dinov3", "dinov3_base"}:
        if not weights:
            raise ValueError("DINOv3 requires a local ViT-B/16 checkpoint path")
        backbone = NativeDinoBackbone(3, weights, freeze=freeze)
        return _configure_adapter(backbone, adapter)
    if key not in aliases:
        raise ValueError(
            f"Unsupported backbone {name!r}. Available: {sorted(aliases)}"
        )
    source = weights or aliases[key]
    backbone = HFVisualBackbone(
        source,
        freeze=freeze,
        local_files_only=local_files_only,
    )
    return _configure_adapter(backbone, adapter)


def _configure_adapter(backbone: nn.Module, adapter: Optional[dict[str, Any]]) -> nn.Module:
    if not adapter or not bool(adapter.get("enabled", False)):
        backbone.adapter_enabled = False
        backbone.adapter_modules = []
        return backbone
    adapter_type = str(adapter.get("type", "lora")).lower()
    if adapter_type != "lora":
        raise ValueError(f"Unsupported adapter type: {adapter_type!r}")
    target_modules = adapter.get("target_modules", ["qkv", "query", "key", "value"])
    matched = inject_lora(
        backbone.model,
        rank=int(adapter.get("rank", 8)),
        alpha=float(adapter.get("alpha", 16.0)),
        dropout=float(adapter.get("dropout", 0.05)),
        target_modules=target_modules,
    )
    backbone.adapter_enabled = True
    backbone.adapter_modules = matched
    return backbone

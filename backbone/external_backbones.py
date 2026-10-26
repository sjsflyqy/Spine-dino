"""Adapters for external image-only checkpoints that are not HF DINO models."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from baseline_pretrain.models import VisionTransformer

from .checkpoints import extract_state_dict, load_checkpoint
from .registry import BackboneOutput


class CXFormerBackbone(nn.Module):
    """Loader for a CXFormer encoder exported by ``baseline_pretrain``."""

    def __init__(self, weights: str, *, freeze: bool = True) -> None:
        super().__init__()
        path = Path(weights).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"CXFormer backbone checkpoint not found: {path}")
        self.model = VisionTransformer(
            dim=768, depth=12, heads=12, image_size=518, patch_size=14,
            registers=4, masked=True, layerscale=True, layerscale_init=1.0,
            separate_qkv=True, gradient_checkpointing=False,
        )
        payload = load_checkpoint(path, allow_legacy_pickle=True)
        state = extract_state_dict(payload, path)
        expected = self.model.state_dict()
        selected = {key: value for key, value in state.items() if key in expected}
        missing = sorted(set(expected) - set(selected))
        unexpected = sorted(set(state) - set(expected))
        shapes = {
            key: (tuple(value.shape), tuple(expected[key].shape))
            for key, value in selected.items() if tuple(value.shape) != tuple(expected[key].shape)
        }
        if missing or unexpected or shapes:
            raise ValueError(
                f"CXFormer checkpoint mismatch: missing={missing[:8]}, "
                f"unexpected={unexpected[:8]}, shapes={dict(list(shapes.items())[:4])}"
            )
        self.model.load_state_dict(selected, strict=True)
        self.patch_size = 14
        self.embedding_dim = 768
        self.model_name = "cxformer_base_export"
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
        cls_token, patch_tokens = self.model(pixel_values)
        grid_h = pixel_values.shape[-2] // self.patch_size
        grid_w = pixel_values.shape[-1] // self.patch_size
        expected = grid_h * grid_w
        if patch_tokens.shape[1] != expected:
            raise RuntimeError(f"Cannot reshape CXFormer tokens to {grid_h}x{grid_w}")
        feature_map = patch_tokens.transpose(1, 2).reshape(
            pixel_values.shape[0], self.embedding_dim, grid_h, grid_w
        )
        return BackboneOutput(cls_token, patch_tokens, feature_map,
                              self.patch_size, self.embedding_dim, (grid_h, grid_w))


class EVAXBackbone(nn.Module):
    """EVA-X-Base image encoder using the authors' timm implementation."""

    def __init__(self, weights: str, *, freeze: bool = True) -> None:
        super().__init__()
        path = Path(weights).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"EVA-X checkpoint not found: {path}")
        try:
            import timm  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "EVA-X requires timm>=0.9.0; install requirements-downstream.txt"
            ) from exc
        source = Path(__file__).resolve().parents[1] / "reference_code/baseline/EVA-X/EVA-X-main"
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        from eva_x import eva_x_base_patch16

        self.model = eva_x_base_patch16(pretrained=str(path))
        self.patch_size = 16
        self.embedding_dim = int(getattr(self.model, "embed_dim", 768))
        self.model_name = "eva_x_base"
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
        # The released EVA-X model is built with a fixed 224x224 patch
        # embedding. Downstream tasks use other image sizes, so adapt the
        # tensor before entering the native model while keeping normalized
        # landmark coordinates unchanged.
        image_size = self.model.patch_embed.img_size
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        if tuple(pixel_values.shape[-2:]) != tuple(image_size):
            pixel_values = F.interpolate(
                pixel_values, size=tuple(image_size), mode="bicubic", align_corners=False
            )
        tokens = self.model.forward_features(pixel_values)
        if tokens.ndim != 3:
            raise RuntimeError(f"Unexpected EVA-X feature shape: {tuple(tokens.shape)}")
        prefix = int(getattr(self.model, "num_prefix_tokens", 1))
        patch_tokens = tokens[:, prefix:]
        grid_h = image_size[0] // self.patch_size
        grid_w = image_size[1] // self.patch_size
        expected = grid_h * grid_w
        if patch_tokens.shape[1] != expected:
            patch_tokens = patch_tokens[:, -expected:]
        if patch_tokens.shape[1] != expected:
            raise RuntimeError(
                f"Cannot reshape EVA-X tokens {patch_tokens.shape[1]} to {grid_h}x{grid_w}"
            )
        cls_token = tokens[:, 0]
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

"""Ray-DINO X-ray ViT-L/16 backbone adapter."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import torch
from torch import Tensor, nn

from .checkpoints import extract_state_dict, load_checkpoint
from .registry import BackboneOutput


class RayDinoBackbone(nn.Module):
    """Load the public Ray-DINO ViT-L/16 encoder.

    The public file is an upstream DINOv2-style state dictionary with four
    ``BlockChunk`` containers.  Trained exports from ``baseline_pretrain``
    contain the same encoder with a flat block numbering, so both formats are
    accepted here.
    """

    def __init__(self, weights: str, *, freeze: bool = True) -> None:
        super().__init__()
        path = Path(weights).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Ray-DINO checkpoint not found: {path}")

        source = Path(__file__).resolve().parents[1] / "upstream" / "dinov2-main"
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        from dinov2.models.vision_transformer import vit_large

        self.model = vit_large(
            patch_size=16,
            img_size=512,
            block_chunks=4,
            num_register_tokens=0,
            init_values=1.0,
        )
        payload = load_checkpoint(path)
        state = extract_state_dict(payload, path)
        expected = self.model.state_dict()
        state = self._normalize_blocks(state, expected)
        selected = {key: value for key, value in state.items() if key in expected}
        shapes = {
            key: (tuple(value.shape), tuple(expected[key].shape))
            for key, value in selected.items()
            if tuple(value.shape) != tuple(expected[key].shape)
        }
        missing = sorted(set(expected) - set(selected))
        unexpected = sorted(set(state) - set(expected))
        if missing or unexpected or shapes:
            raise ValueError(
                f"Ray-DINO checkpoint mismatch: missing={missing[:8]}, "
                f"unexpected={unexpected[:8]}, shapes={dict(list(shapes.items())[:4])}"
            )
        self.model.load_state_dict(selected, strict=True)
        self.patch_size = 16
        self.embedding_dim = 1024
        self.model_name = "ray_dino_vitl16"
        if freeze:
            self.freeze()

    @staticmethod
    def _normalize_blocks(state, expected):
        normalized = {}
        for key, value in state.items():
            # Official Ray-DINO uses blocks.<chunk>.<global_index>.<suffix>.
            # Exports from baseline_pretrain use blocks.<global_index>.<suffix>.
            # Restore the latter to the upstream chunked layout.
            if key not in expected:
                match = re.match(r"^blocks\.(\d+)\.(.+)$", key)
                if match:
                    index, suffix = int(match.group(1)), match.group(2)
                    key = f"blocks.{index // 6}.{index}.{suffix}"
            if key.startswith(("dino_head.", "ibot_head.")):
                continue
            normalized[key] = value
        return normalized

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
        height, width = pixel_values.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("Ray-DINO input height/width must be divisible by patch size 16")
        output = self.model.forward_features(pixel_values)
        cls_token = output["x_norm_clstoken"]
        patch_tokens = output["x_norm_patchtokens"]
        grid_h, grid_w = height // self.patch_size, width // self.patch_size
        expected = grid_h * grid_w
        if patch_tokens.shape[1] != expected:
            raise RuntimeError(
                f"Cannot reshape Ray-DINO tokens {patch_tokens.shape[1]} to {grid_h}x{grid_w}"
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

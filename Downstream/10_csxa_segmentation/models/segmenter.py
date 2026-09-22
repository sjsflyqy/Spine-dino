"""Frozen/LoRA DINO encoder with linear or SpineFM SAM mask decoder."""
from __future__ import annotations

import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from common import REPO, resolve

sys.path.insert(0, str(REPO))
from backbone import build_backbone
from .sam_parts import MaskDecoder, PromptEncoder, TwoWayTransformer


class Segmenter(nn.Module):
    def __init__(self, config, backbone=None):
        super().__init__()
        self.config = config
        self.mode = config.get("mode", "decoder")
        if self.mode not in {"linear", "decoder", "lora", "lora_linear"}:
            raise ValueError(self.mode)
        self.uses_lora = self.mode in {"lora", "lora_linear"}
        self.uses_decoder = self.mode in {"decoder", "lora"}
        self.input_size = int(config.get("input_size", 896))
        self.target_size = int(config.get("target_size", 256))
        adapter = config.get("adapter") if self.uses_lora else None
        if self.uses_lora and not (adapter and adapter.get("enabled")):
            raise ValueError("LoRA mode requires model.adapter.enabled=true")
        self.backbone = backbone if backbone is not None else build_backbone(
            config["backbone"], str(resolve(config["weights"])), freeze=True,
            local_files_only=True, adapter=adapter)
        if self.uses_lora and config.get("gradient_checkpointing", True):
            encoder = self.backbone.model
            blocks = getattr(encoder, "blocks", None)
            if blocks is None:
                blocks = getattr(getattr(encoder, "encoder", None), "layer", None)
            if blocks is None:
                raise ValueError("Cannot locate transformer blocks for gradient checkpointing")
            # Wrap forwards without renaming state-dict keys or LoRA parameters.
            for block in blocks:
                block.forward = self._checkpoint_forward(block.forward)
        if self.input_size % self.backbone.patch_size:
            raise ValueError("input_size must be divisible by backbone patch_size")
        dim = self.backbone.embedding_dim
        # Backbone construction/LoRA injection consumes a different number of random
        # draws across architectures. Reset here to make downstream initialization equal.
        torch.manual_seed(int(config.get("head_seed", 42)))
        if not self.uses_decoder:
            self.head = nn.Conv2d(dim, 1, 1)
        else:
            self.grid = int(config.get("embedding_grid", 64))
            self.feature_adapter = nn.Sequential(nn.Conv2d(dim, 256, 1), nn.GroupNorm(1, 256))
            self.prompt_encoder = PromptEncoder(256, (self.grid, self.grid),
                                                (self.input_size, self.input_size), 16)
            self.mask_decoder = MaskDecoder(transformer_dim=256, num_multimask_outputs=1,
                                            transformer=TwoWayTransformer(depth=2, embedding_dim=256,
                                                                          num_heads=8, mlp_dim=2048),
                                            iou_head_depth=3, iou_head_hidden_dim=256)
            if config.get("sam_initialization"):
                self.load_sam_initialization(resolve(config["sam_initialization"]))

    def _checkpoint_forward(self, forward):
        def wrapped(*args, **kwargs):
            if self.training and torch.is_grad_enabled():
                return checkpoint(forward, *args, use_reentrant=False, **kwargs)
            return forward(*args, **kwargs)
        return wrapped

    def load_sam_initialization(self, path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        state = state.get("state_dict", state)
        # Decoder-only import; never replace the evaluated image encoder.
        for name in ("prompt_encoder", "mask_decoder"):
            module = getattr(self, name)
            selected = {k[len(name)+1:]: v for k, v in state.items() if k.startswith(name + ".")}
            # Official SAM uses lin1/lin2; SpineFM names the same two affine
            # maps layers.0.0/fc. Map these known equivalent keys, then require
            # every parameter/buffer to load, rather than silently dropping MLPs.
            if name == "mask_decoder":
                selected = {k.replace(".mlp.lin1.", ".mlp.layers.0.0.")
                             .replace(".mlp.lin2.", ".mlp.fc."): v for k, v in selected.items()}
            module.load_state_dict(selected, strict=True)

    def train(self, mode=True):
        super().train(mode)
        if not getattr(self.backbone, "adapter_enabled", False):
            self.backbone.eval()
        return self

    def forward(self, images, points):
        context = torch.enable_grad() if self.training and getattr(self.backbone, "adapter_enabled", False) else torch.no_grad()
        with context:
            features = self.backbone(images).feature_map
        if not self.uses_decoder:
            logits, quality = self.head(features), None
        else:
            embeddings = F.interpolate(self.feature_adapter(features), (self.grid, self.grid),
                                       mode="bilinear", align_corners=False)
            sparse, dense = self.prompt_encoder(
                points=(points[:, None], torch.ones((len(points), 1), device=points.device, dtype=torch.int64)),
                boxes=None, masks=None)
            logits, quality = self.mask_decoder(embeddings, self.prompt_encoder.get_dense_pe(),
                                                 sparse, dense, multimask_output=False)
        logits = F.interpolate(logits, (self.target_size, self.target_size), mode="bilinear", align_corners=False)
        return logits, quality


def segmentation_loss(logits, targets, quality=None):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probabilities = logits.sigmoid()
    intersection = (probabilities * targets).sum((1, 2, 3))
    dice = 1 - ((2*intersection + 1) / (probabilities.sum((1, 2, 3)) + targets.sum((1, 2, 3)) + 1)).mean()
    loss = bce + dice
    if quality is not None:
        with torch.no_grad():
            pred, gt = probabilities > 0.5, targets > 0.5
            iou = (pred & gt).sum((1, 2, 3)).float() / (pred | gt).sum((1, 2, 3)).clamp_min(1)
        loss = loss + 0.1 * F.mse_loss(quality[:, 0], iou)
    return loss

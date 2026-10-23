# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0; see third-party notices.
"""Checkpoint-compatible ViTs using PyTorch SDPA, without timm/global import hooks.

Adapted from Meta's DINO/DINOv2/MAE and the local DAX, CheXFound and Medical
MAE references. See THIRD_PARTY_NOTICES.md and LICENSE-APACHE-2.0.txt.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class PatchEmbed(nn.Module):
    def __init__(self, dim, patch_size=16):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, patch_size, patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        return self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, d))


class Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden = (int(4 * dim * 2 / 3) + 7) // 8 * 8
        self.w12 = nn.Linear(dim, 2 * hidden)
        self.w3 = nn.Linear(hidden, dim)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, -1)
        return self.w3(F.silu(a) * b)


class LayerScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), 1e-5))

    def forward(self, x):
        return x * self.gamma


class Block(nn.Module):
    def __init__(self, dim, heads, *, swiglu=False, layerscale=False, drop_path=0.0, subset_drop=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = SwiGLU(dim) if swiglu else Mlp(dim, 4 * dim)
        self.ls1 = LayerScale(dim) if layerscale else nn.Identity()
        self.ls2 = LayerScale(dim) if layerscale else nn.Identity()
        self.drop_path, self.subset_drop = drop_path, subset_drop

    def residual(self, x, fn):
        if not self.training or not self.drop_path:
            return x + fn(x)
        if self.subset_drop and self.drop_path > 0.1:
            # Match DINOv2/CheXFound's high-drop-rate subset stochastic depth.
            count = max(int(len(x) * (1 - self.drop_path)), 1)
            indices = torch.randperm(len(x), device=x.device)[:count]
            return torch.index_add(x, 0, indices, fn(x[indices]).to(x.dtype), alpha=len(x) / count)
        keep = 1 - self.drop_path
        mask = x.new_empty((len(x), 1, 1)).bernoulli_(keep) / keep
        return x + fn(x) * mask

    def forward(self, x):
        x = self.residual(x, lambda y: self.ls1(self.attn(self.norm1(y))))
        return self.residual(x, lambda y: self.ls2(self.mlp(self.norm2(y))))


class VisionTransformer(nn.Module):
    def __init__(self, *, dim, depth, heads, image_size, patch_size=16, registers=0,
                 swiglu=False, layerscale=False, drop_path=0.0, drop_path_uniform=False,
                 antialias=False, pos_offset=0.1, gradient_checkpointing=False, masked=False):
        super().__init__()
        self.dim, self.depth, self.patch_size = dim, depth, patch_size
        self.register_count, self.antialias, self.pos_offset = registers, antialias, pos_offset
        self.gradient_checkpointing = gradient_checkpointing
        self.patch_embed = PatchEmbed(dim, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, (image_size // patch_size) ** 2 + 1, dim))
        if registers:
            self.register_tokens = nn.Parameter(torch.zeros(1, registers, dim))
        if masked:
            self.mask_token = nn.Parameter(torch.zeros(1, dim))
        self.blocks = nn.ModuleList([
            Block(dim, heads, swiglu=swiglu, layerscale=layerscale, subset_drop=layerscale,
                  drop_path=drop_path if drop_path_uniform else drop_path * i / max(depth - 1, 1))
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def position(self, h, w):
        gh, gw = h // self.patch_size, w // self.patch_size
        side = math.isqrt(self.pos_embed.shape[1] - 1)
        if gh == gw == side:
            return self.pos_embed
        p = self.pos_embed[:, 1:].float().reshape(1, side, side, self.dim).permute(0, 3, 1, 2)
        kwargs = {"scale_factor": ((gh + self.pos_offset) / side, (gw + self.pos_offset) / side)} if self.pos_offset else {"size": (gh, gw)}
        p = F.interpolate(p, mode="bicubic", align_corners=False, antialias=self.antialias, **kwargs)
        return torch.cat((self.pos_embed[:, :1], p.flatten(2).transpose(1, 2)), 1).to(self.pos_embed.dtype)

    def run_blocks(self, x):
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.norm(x)

    def forward(self, images, masks=None):
        h, w = images.shape[-2:]
        if h % self.patch_size or w % self.patch_size:
            raise ValueError("Image height/width must be divisible by patch size")
        x = self.patch_embed(images)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
        x = torch.cat((self.cls_token.expand(len(x), -1, -1), x), 1) + self.position(h, w)
        if self.register_count:
            x = torch.cat((x[:, :1], self.register_tokens.expand(len(x), -1, -1), x[:, 1:]), 1)
        x = self.run_blocks(x)
        return x[:, 0], x[:, 1 + self.register_count:]


class WeightNormLinear(nn.Module):
    """Explicit legacy-compatible weight normalization, safe inside FSDP."""
    def __init__(self, dim, out):
        super().__init__()
        self.weight_v = nn.Parameter(torch.empty(out, dim))
        self.weight_g = nn.Parameter(torch.ones(out, 1))
        nn.init.normal_(self.weight_v, std=0.02)

    def forward(self, x):
        weight = F.normalize(self.weight_v, dim=1, eps=1e-12) * self.weight_g
        return F.linear(x, weight)


class DINOHead(nn.Module):
    def __init__(self, dim, out, hidden=2048, bottleneck=256, freeze_gain=False):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, hidden),
                                 nn.GELU(), nn.Linear(hidden, bottleneck))
        self.last_layer = WeightNormLinear(bottleneck, out)
        self.last_layer.weight_g.requires_grad_(not freeze_gain)

    def forward(self, x):
        x = self.mlp(x)
        return self.last_layer(F.normalize(x, dim=-1, eps=1e-6 if x.dtype == torch.float16 else 1e-12))


class DistillationNetwork(nn.Module):
    def __init__(self, method, model, objective, *, teacher=False):
        super().__init__()
        spec = dict(model)
        if teacher:
            spec.update(drop_path=0.0, gradient_checkpointing=False)
        self.method = method
        self.is_teacher = teacher
        self.backbone = VisionTransformer(**spec)
        self.dino_head = DINOHead(spec["dim"], objective["prototypes"], objective["head_hidden"],
                                  objective["dino_bottleneck"], objective.get("freeze_gain", False))
        if method == "chexfound":
            self.ibot_head = DINOHead(spec["dim"], objective["prototypes"], objective["head_hidden"], objective["ibot_bottleneck"])

    def forward(self, crops, masks=None):
        globals_ = torch.cat(crops[:2])
        cls, patch = self.backbone(globals_, None if self.is_teacher else masks)
        all_cls = cls
        if len(crops) > 2:
            local_cls, _ = self.backbone(torch.cat(crops[2:]))
            all_cls = torch.cat((cls, local_cls))
        out = {"cls_logits": self.dino_head(all_cls), "global_cls": cls}
        if self.method == "chexfound":
            # Empty masking is rejected by the objective; every rank executes the same head path.
            out["patch_logits"] = self.ibot_head(patch[masks])
        return out


class MaskedAutoencoder(VisionTransformer):
    def __init__(self, model, decoder):
        super().__init__(**model)
        self.pos_embed.requires_grad_(False)
        d = decoder["dim"]
        self.decoder_embed = nn.Linear(self.dim, d)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.pos_embed.shape[1], d), requires_grad=False)
        self.decoder_blocks = nn.ModuleList([Block(d, decoder["heads"]) for _ in range(decoder["depth"])])
        self.decoder_norm = nn.LayerNorm(d, eps=1e-6)
        self.decoder_pred = nn.Linear(d, self.patch_size ** 2 * 3)

    def forward(self, images, mask_ratio=0.9, norm_pix_loss=False):
        x = self.patch_embed(images) + self.pos_embed[:, 1:]
        b, n, d = x.shape
        keep = int(n * (1 - mask_ratio))
        if not 0 < keep < n:
            raise ValueError("mask_ratio must leave at least one visible and one masked patch")
        ids = torch.argsort(torch.rand(b, n, device=x.device), dim=1)
        restore = ids.argsort(1)
        x = x.gather(1, ids[:, :keep, None].expand(-1, -1, d))
        cls = (self.cls_token + self.pos_embed[:, :1]).expand(b, -1, -1)
        x = self.decoder_embed(self.run_blocks(torch.cat((cls, x), 1)))
        patches = torch.cat((x[:, 1:], self.mask_token.expand(b, n - keep, -1)), 1)
        patches = patches.gather(1, restore[:, :, None].expand(-1, -1, patches.shape[-1]))
        x = torch.cat((x[:, :1], patches), 1) + self.decoder_pos_embed
        for block in self.decoder_blocks:
            x = checkpoint(block, x, use_reentrant=False) if self.gradient_checkpointing and self.training else block(x)
        prediction = self.decoder_pred(self.decoder_norm(x))[:, 1:]
        p = self.patch_size
        h, w = images.shape[-2] // p, images.shape[-1] // p
        target = images.reshape(b, 3, h, p, w, p).permute(0, 2, 4, 3, 5, 1).reshape(b, n, p * p * 3)
        if norm_pix_loss:
            target = (target - target.mean(-1, keepdim=True)) / (target.var(-1, keepdim=True) + 1e-6).sqrt()
        mask = torch.ones(b, n, device=x.device)
        mask[:, :keep] = 0
        mask = mask.gather(1, restore)
        per_patch = (prediction.float() - target.float()).square().mean(-1)
        return (per_patch * mask).sum() / mask.sum()


def build_network(config, *, teacher=False, device=None):
    with torch.device(device or "cpu"):
        if config["method"] == "medical_mae":
            return MaskedAutoencoder(config["model"], config["decoder"])
        return DistillationNetwork(config["method"], config["model"], config["objective"], teacher=teacher)

"""Small MAE-style decoder for the full, ordered iBOT patch sequence.

No encoder masking, token insertion, or sequence restoration happens here.
The encoder has already contextualized both visible and masked positions.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def sincos_position_embedding(height: int, width: int, dim: int) -> torch.Tensor:
    """Fixed 2-D encoding in row-major patch order, without CLS/registers."""
    if min(height, width, dim) <= 0 or dim % 4:
        raise ValueError("grid sizes must be positive and decoder_dim divisible by 4")
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    frequency = 10000.0 ** (-torch.arange(dim // 4, dtype=torch.float32) / (dim // 4))
    x = x.flatten().float()[:, None] * frequency
    y = y.flatten().float()[:, None] * frequency
    return torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=-1).unsqueeze(0)


class PixelDecoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(batch, length, 3, self.num_heads, dim // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        x = x + self.proj(attention.transpose(1, 2).reshape(batch, length, dim))
        return x + self.mlp(self.norm2(x))


class PixelDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        grid_size: tuple[int, int],
        patch_size: int,
        in_chans: int = 3,
        decoder_dim: int = 256,
        decoder_depth: int = 2,
        decoder_num_heads: int = 8,
    ):
        super().__init__()
        if min(embed_dim, patch_size, in_chans, decoder_dim, decoder_depth, decoder_num_heads) <= 0:
            raise ValueError("pixel decoder dimensions, depth, and heads must be positive")
        if decoder_dim % decoder_num_heads:
            raise ValueError("decoder_dim must be divisible by decoder_num_heads")
        self.grid_size = tuple(grid_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.decoder_embed = nn.Linear(embed_dim, decoder_dim)
        self.register_buffer("pos_embed", sincos_position_embedding(*grid_size, decoder_dim))
        self.layers = nn.ModuleList([
            PixelDecoderBlock(decoder_dim, decoder_num_heads) for _ in range(decoder_depth)
        ])
        self.norm = nn.LayerNorm(decoder_dim, eps=1e-6)
        self.pred = nn.Linear(decoder_dim, patch_size * patch_size * in_chans)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        if patch_tokens.ndim != 3 or patch_tokens.shape[1] != self.pos_embed.shape[1]:
            raise ValueError("pixel decoder requires the full patch grid, excluding CLS/register tokens")
        x = self.decoder_embed(patch_tokens)
        x = x + self.pos_embed.to(dtype=x.dtype)
        for layer in self.layers:
            x = layer(x)
        return self.pred(self.norm(x))

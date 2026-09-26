"""
R-Build vision tower (v2.1) — complete VL support.

A ViT-style encoder turns images (or sampled video frames) into soft tokens
projected to `d_model`. Those soft tokens are spliced into the token stream
at reserved `<image>` placeholder positions, so the cache-loop line and the
parallel bundle stages see a single unified sequence — vision is just more
tokens. Video is frames-as-token-groups: each frame is encoded independently
and gets a learned frame-position embedding, giving whole-video understanding
inside the same sequence.

Everything is user-modifiable via `VisionConfig`; with `enabled=False`
(the default — "blind" mode) the tower is not built at all and the model is
bit-identical to the text-only v2.0 path.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.attention import RMSNorm


class _ViTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ffn_mult: float = 4.0, eps: float = 1e-6):
        super().__init__()
        self.norm1 = RMSNorm(dim, eps)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, bias=False)
        self.norm2 = RMSNorm(dim, eps)
        ffn = int(dim * ffn_mult)
        self.w1 = nn.Linear(dim, ffn, bias=False)
        self.w2 = nn.Linear(ffn, dim, bias=False)
        self.w3 = nn.Linear(dim, ffn, bias=False)

    def forward(self, x):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        h = self.norm2(x)
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h))


class VisionTower(nn.Module):
    """
    Patchify -> ViT blocks -> project to d_model.

    Input:  images (B, N, C, H, W)  — N = images per sample (1) or video frames
    Output: (B, N * tokens_per_image, d_model) soft tokens, with a learned
            frame-position embedding added when N > 1 (video).
    """

    def __init__(self, cfg):
        super().__init__()
        v = cfg.vision
        d = cfg.model.d_model
        self.image_size = v.image_size
        self.patch_size = v.patch_size
        self.grid = v.image_size // v.patch_size
        self.tokens_per_image = self.grid * self.grid
        vit_dim = v.vit_dim or d

        self.patch_proj = nn.Conv2d(v.channels, vit_dim, kernel_size=v.patch_size,
                                    stride=v.patch_size, bias=False)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.tokens_per_image, vit_dim))
        self.cls = nn.Parameter(torch.zeros(1, 1, vit_dim)) if v.use_cls_token else None
        if v.use_cls_token:
            self.tokens_per_image += 1
            self.pos_embed = nn.Parameter(torch.zeros(1, self.tokens_per_image, vit_dim))
        self.blocks = nn.ModuleList(
            _ViTBlock(vit_dim, v.vit_heads, v.vit_ffn_mult, cfg.model.rmsnorm_eps)
            for _ in range(v.vit_layers)
        )
        self.out_norm = RMSNorm(vit_dim, cfg.model.rmsnorm_eps)
        self.projector = nn.Linear(vit_dim, d, bias=False)
        # video: learned per-frame position, max frames user-set
        self.frame_embed = nn.Embedding(v.max_video_frames, d) if v.video else None
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.projector.weight, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = images.shape
        assert H == self.image_size and W == self.image_size, \
            f"expected {self.image_size}px inputs, got {H}x{W} — resize/crop upstream"
        x = images.reshape(B * N, C, H, W)
        x = self.patch_proj(x)                            # (B*N, vit_dim, g, g)
        x = x.flatten(2).transpose(1, 2)                  # (B*N, g*g, vit_dim)
        if self.cls is not None:
            x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.projector(self.out_norm(x))              # (B*N, T_img, d_model)
        x = x.reshape(B, N * self.tokens_per_image, -1)
        if self.frame_embed is not None and N > 1:
            assert N <= self.frame_embed.num_embeddings, \
                f"{N} frames > max_video_frames={self.frame_embed.num_embeddings}"
            frame_pos = self.frame_embed.weight[:N]         # (N, d)
            frame_pos = frame_pos.repeat_interleave(self.tokens_per_image, dim=0)
            x = x + frame_pos.unsqueeze(0)
        return x

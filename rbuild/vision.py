"""
R-Build vision (v2.1 ViT tower + v3 encoderless mode and VaWU).

Three ways to see, all user-modifiable via `VisionConfig`:

  mode="vit"         (v2.1 default when enabled) — a ViT tower encodes
                     images / sampled video frames into soft tokens.

  mode="encoderless" (v3) — *no vision encoder at all*. Patches are
                     normalized and linearly projected straight into
                     d_model, then handed to the same cache-loop and
                     parallel stages as text. VL with zero vision-specific
                     compute path: the LLM itself is the vision encoder.
                     VaWU pairs naturally with this.

  vawu=True          (v3) — Video-as-Whole-Understanding: after per-frame
                     encoding (either mode), a learned query attention-pools
                     all frames into whole-video summary tokens that are
                     prepended to the frame stream, so the model
                     always reads the video *as a whole* before the parts.

Default remains blind: `enabled=False` builds nothing and the model is
bit-identical to the text-only path.
"""

from __future__ import annotations

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


class VaWUPooler(nn.Module):
    """
    Video-as-Whole-Understanding: attention-pool every frame's tokens into
    a small set of whole-video summary tokens via learned queries. The
    summary tokens are prepended to the frame stream, so downstream stages
    read the whole video before its frames.
    """

    def __init__(self, d_model: int, n_tokens: int, n_heads: int = 8, eps: float = 1e-6):
        super().__init__()
        self.n_tokens = n_tokens
        self.queries = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, bias=False)
        self.norm = RMSNorm(d_model, eps)
        nn.init.normal_(self.queries, std=0.02)

    def forward(self, frame_tokens: torch.Tensor) -> torch.Tensor:
        """frame_tokens: (B, N*T_img, D) -> (B, n_tokens, D) whole-video tokens."""
        B = frame_tokens.shape[0]
        q = self.queries.expand(B, -1, -1)
        out, _ = self.attn(q, frame_tokens, frame_tokens, need_weights=False)
        return self.norm(out + q)


class EncoderlessScreenshotProjector(nn.Module):
    """Single-weight patch projector for adapted-model CUA screenshots.

    This follows the encoderless vision path without constructing a second
    vision encoder. It is used only by the CUA interface and never rewrites an
    adapted base model's native forward/generation computation.
    """

    def __init__(self, hidden_size: int, image_size: int = 64,
                 patch_size: int = 16, channels: int = 3):
        super().__init__()
        if hidden_size <= 0 or image_size <= 0 or patch_size <= 0 or channels <= 0:
            raise ValueError("Screenshot projector dimensions must be positive")
        if image_size % patch_size:
            raise ValueError("Screenshot image_size must be divisible by patch_size")
        self.hidden_size = int(hidden_size)
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.channels = int(channels)
        self.patch_proj = nn.Conv2d(
            channels, hidden_size, kernel_size=patch_size, stride=patch_size, bias=False
        )

    def forward(self, screenshots: torch.Tensor) -> torch.Tensor:
        if screenshots.ndim == 4:
            screenshots = screenshots.unsqueeze(1)
        if screenshots.ndim != 5:
            raise ValueError("Screenshots must have shape (B,C,H,W) or (B,N,C,H,W)")
        batch, frames, channels, height, width = screenshots.shape
        if channels != self.channels or height != self.image_size or width != self.image_size:
            raise ValueError(
                f"Expected {self.channels}x{self.image_size}x{self.image_size} screenshots"
            )
        patches = self.patch_proj(screenshots.reshape(batch * frames, channels, height, width))
        tokens = patches.flatten(2).transpose(1, 2)
        return tokens.reshape(batch, frames * tokens.shape[1], self.hidden_size)


class VisionTower(nn.Module):
    """
    Two modes, one interface:

      vit         : patchify -> ViT blocks -> project to d_model
      encoderless : patchify -> linear to d_model (no encoder at all)

    Input:  images (B, N, C, H, W)  — N = images per sample (1) or video frames
    Output: (B, N * tokens_per_image [+ vawu_tokens], d_model) soft tokens,
            with learned frame-position embeddings when N > 1, and VaWU
            whole-video tokens prepended when vawu=True and N > 1.
    """

    def __init__(self, cfg):
        super().__init__()
        v = cfg.vision
        d = cfg.model.d_model
        self.mode = v.mode
        self.image_size = v.image_size
        self.patch_size = v.patch_size
        self.grid = v.image_size // v.patch_size
        self.tokens_per_image = self.grid * self.grid
        vit_dim = v.vit_dim or d

        # patch embedding is shared by both modes
        self.patch_proj = nn.Conv2d(v.channels, vit_dim if self.mode == "vit" else d,
                                    kernel_size=v.patch_size,
                                    stride=v.patch_size, bias=False)
        if self.mode == "vit":
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
            nn.init.normal_(self.pos_embed, std=0.02)
            nn.init.normal_(self.projector.weight, std=0.02)
        else:
            # encoderless: patch norm + the LLM does the rest (no encoder params)
            self.pos_embed = nn.Parameter(torch.zeros(1, self.tokens_per_image, d))
            self.patch_norm = RMSNorm(d, cfg.model.rmsnorm_eps)
            nn.init.normal_(self.pos_embed, std=0.02)

        # video: learned per-frame position, max frames user-set
        self.frame_embed = nn.Embedding(v.max_video_frames, d) if v.video else None
        # VaWU whole-video summary tokens (prepended when N > 1)
        if v.vawu and v.video:
            want = max(1, v.vit_heads // 2)
            heads = max(h for h in range(1, want + 1) if d % h == 0)
            self.vawu = VaWUPooler(d, v.vawu_tokens, n_heads=heads,
                                   eps=cfg.model.rmsnorm_eps)
        else:
            self.vawu = None

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = images.shape
        assert H == self.image_size and W == self.image_size, \
            f"expected {self.image_size}px inputs, got {H}x{W} — resize/crop upstream"
        x = images.reshape(B * N, C, H, W)
        x = self.patch_proj(x)                            # (B*N, ·, g, g)
        x = x.flatten(2).transpose(1, 2)                  # (B*N, g*g, ·)
        if self.mode == "vit":
            if self.cls is not None:
                x = torch.cat([self.cls.expand(x.shape[0], -1, -1)], dim=1)
            x = x + self.pos_embed
            for blk in self.blocks:
                x = blk(x)
            x = self.projector(self.out_norm(x))          # (B*N, T_img, d_model)
        else:
            x = self.patch_norm(x + self.pos_embed)       # (B*N, T_img, d_model)
        x = x.reshape(B, N * self.tokens_per_image, -1)

        whole = None
        if self.frame_embed is not None and N > 1:
            assert N <= self.frame_embed.num_embeddings, \
                f"{N} frames > max_video_frames={self.frame_embed.num_embeddings}"
            frame_pos = self.frame_embed.weight[:N]         # (N, d)
            frame_pos = frame_pos.repeat_interleave(self.tokens_per_image, dim=0)
            x = x + frame_pos.unsqueeze(0)
            if self.vawu is not None:
                whole = self.vawu(x)                        # (B, vawu_tokens, D)
        elif self.vawu is not None:
            whole = self.vawu(x)
        if whole is not None:
            x = torch.cat([whole, x], dim=1)                # whole first, then parts
        return x

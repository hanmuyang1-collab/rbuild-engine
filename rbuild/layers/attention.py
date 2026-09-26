"""
Causal attention with GQA + RoPE, an inference KV cache, and a Mixture-of-
Depths router (only the top-p tokens are computed by the block; the rest
pass through the residual stream untouched — the R-Compute lever).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


def _rope_cos_sin(seq_len: int, head_dim: int, theta: float, device, dtype):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv)                       # (T, hd/2)
    cos = freqs.cos()[None, :, None, :].to(dtype)     # (1, T, 1, hd/2)
    sin = freqs.sin()[None, :, None, :].to(dtype)
    return cos, sin


def _apply_rope(x: torch.Tensor, cos, sin) -> torch.Tensor:
    # x: (B, T, H, hd)
    x1, x2 = x.chunk(2, dim=-1)
    T = x.shape[1]
    return torch.cat([x1 * cos[:, :T] - x2 * sin[:, :T],
                      x2 * cos[:, :T] + x1 * sin[:, :T]], dim=-1)


class CausalAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int,
                 head_dim: Optional[int] = None, rope_theta: float = 10000.0,
                 max_seq_len: int = 2048):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim or d_model // n_heads
        self.rope_theta = rope_theta
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        self.max_seq_len = max_seq_len

    def forward(
        self,
        x: torch.Tensor,                       # (B, T, D)
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        start_pos: int = 0,
        use_cache: bool = False,
    ):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)

        total = start_pos + T
        cos, sin = _rope_cos_sin(total, self.head_dim, self.rope_theta, x.device, x.dtype)
        # offset positions for cached decoding
        cos = cos[:, start_pos:]; sin = sin[:, start_pos:]
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=1)
            v = torch.cat([kv_cache[1], v], dim=1)
        new_cache = (k, v) if use_cache else None

        # GQA: expand kv heads
        rep = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(rep, dim=2)
        v = v.repeat_interleave(rep, dim=2)

        # (B, H, T, hd)
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
        is_causal = kv_cache is None or kv_cache[0].shape[1] == 0
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal and T > 1)
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(out), new_cache


class MoDRouter(nn.Module):
    """
    Mixture-of-Depths router: scores each token, lets the top-`capacity`
    fraction through the block, and skips the rest (they stay in the
    residual stream). Fully user-tunable via `capacity`.
    """

    def __init__(self, d_model: int, capacity: float = 0.5):
        super().__init__()
        self.capacity = capacity
        self.score = nn.Linear(d_model, 1, bias=False)

    def forward(self, x: torch.Tensor, block: nn.Module, **block_kwargs):
        """
        x: (B, T, D); block: callable on (k_tokens, D) batches.
        Returns the updated sequence with only routed tokens replaced.
        """
        B, T, D = x.shape
        k = max(1, int(T * self.capacity))
        scores = self.score(x).squeeze(-1)                  # (B, T) — grad-enabled
        with torch.no_grad():
            top_idx = scores.topk(k, dim=1).indices         # discrete selection
        gathered = torch.gather(x, 1, top_idx.unsqueeze(-1).expand(-1, -1, D))
        # run the block on the flattened routed tokens (batch dims merged)
        out = block(gathered.reshape(B * k, 1, D), **block_kwargs)
        if isinstance(out, tuple):
            out = out[0]
        out = out.reshape(B, k, D)
        # MoD update: x' = x + sigmoid(router_score) * (block(x) - x) for the
        # selected tokens — keeps residual semantics and gives the router a
        # gradient path through the selected outputs.
        sel_scores = torch.gather(scores, 1, top_idx)       # (B, k)
        out = gathered + torch.sigmoid(sel_scores).unsqueeze(-1) * (out - gathered)
        x = x.clone()
        x.scatter_(1, top_idx.unsqueeze(-1).expand(-1, -1, D), out)
        return x

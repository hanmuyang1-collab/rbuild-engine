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


_HAS_FUSED_RMS = hasattr(F, "rms_norm")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # v4.3: one fused kernel (F.rms_norm, torch>=2.4) instead of four
        # elementwise launches. The statistic is still accumulated in fp32.
        if _HAS_FUSED_RMS and x.dtype == torch.float32:
            return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


# v4.3: RoPE tables are cached per (head_dim, theta, device, dtype) and grown
# on demand. v4.2 rebuilt the full (T, hd/2) cos/sin table in every
# attention call of every block of every loop.
_ROPE_CACHE: dict = {}


def _rope_table(length: int, head_dim: int, theta: float, device, dtype):
    key = (head_dim, float(theta), str(device), dtype)
    hit = _ROPE_CACHE.get(key)
    if hit is None or hit[0].shape[1] < length:
        size = max(length, 2 * (hit[0].shape[1] if hit is not None else 0), 256)
        inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device,
                                            dtype=torch.float32) / head_dim))
        t = torch.arange(size, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv)
        hit = (freqs.cos()[None, :, None, :].to(dtype),
               freqs.sin()[None, :, None, :].to(dtype))
        _ROPE_CACHE[key] = hit
    return hit


def _rope_cos_sin(seq_len: int, head_dim: int, theta: float, device, dtype):
    cos, sin = _rope_table(seq_len, head_dim, theta, device, dtype)
    return cos[:, :seq_len], sin[:, :seq_len]


def _rope_at(positions: torch.Tensor, length: int, head_dim: int, theta: float, dtype):
    """cos/sin gathered at explicit positions: (B, T) -> (B, T, 1, hd/2).
    `length` bounds the positions (no device->host sync to discover it)."""
    cos, sin = _rope_table(length, head_dim, theta, positions.device, dtype)
    return cos[0, positions], sin[0, positions]


def _apply_rope(x: torch.Tensor, cos, sin) -> torch.Tensor:
    # x: (B, T, H, hd)
    x1, x2 = x.chunk(2, dim=-1)
    T = x.shape[1]
    return torch.cat([x1 * cos[:, :T] - x2 * sin[:, :T],
                      x2 * cos[:, :T] + x1 * sin[:, :T]], dim=-1)


def _sdpa_supports_gqa() -> bool:
    try:
        q = torch.zeros(1, 2, 1, 2)
        kv = torch.zeros(1, 1, 1, 2)
        F.scaled_dot_product_attention(q, kv, kv, enable_gqa=True)
        return True
    except TypeError:
        return False
    except Exception:  # pragma: no cover - exotic backends
        return False


_SDPA_GQA = _sdpa_supports_gqa()


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
        positions: Optional[torch.Tensor] = None,   # (B, T) original positions
        pos_limit: Optional[int] = None,            # exclusive bound on positions
    ):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)

        if positions is not None:
            # v4.3 MoD: a routed token subset keeps its true positions.
            cos, sin = _rope_at(positions, pos_limit or self.max_seq_len,
                                self.head_dim, self.rope_theta, x.dtype)
        else:
            cos, sin = _rope_table(start_pos + T, self.head_dim, self.rope_theta,
                                   x.device, x.dtype)
            cos = cos[:, start_pos:start_pos + T]; sin = sin[:, start_pos:start_pos + T]
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=1)
            v = torch.cat([kv_cache[1], v], dim=1)
        new_cache = (k, v) if use_cache else None

        # (B, H, T, hd)
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
        is_causal = (kv_cache is None or kv_cache[0].shape[1] == 0) and T > 1
        rep = self.n_heads // self.n_kv_heads
        if rep > 1 and _SDPA_GQA:
            # v4.3: native GQA — no repeat_interleave copy of K/V per call.
            out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal,
                                                 enable_gqa=True)
        else:
            if rep > 1:
                k = k.repeat_interleave(rep, dim=1)
                v = v.repeat_interleave(rep, dim=1)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
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
        x: (B, T, D); block: callable on (B, k, D) token subsets that accepts
        a `positions=` (B, k) keyword.

        v4.3 fix: the routed tokens are processed as ONE causal sequence in
        their original order and at their original RoPE positions. v4.2
        flattened them into B*k independent length-1 sequences, so attention
        inside every MoD-routed block could only see the token itself — the
        blocks were effectively attention-free during training.
        """
        B, T, D = x.shape
        k = max(1, int(T * self.capacity))
        scores = self.score(x).squeeze(-1)                  # (B, T) — grad-enabled
        with torch.no_grad():
            top_idx = scores.topk(k, dim=1).indices         # discrete selection
            top_idx, _ = top_idx.sort(dim=1)                # keep causal order
        gathered = torch.gather(x, 1, top_idx.unsqueeze(-1).expand(-1, -1, D))
        out = block(gathered, positions=top_idx, pos_limit=T, **block_kwargs)
        if isinstance(out, tuple):
            out = out[0]
        # MoD update: x' = x + sigmoid(router_score) * (block(x) - x) for the
        # selected tokens — keeps residual semantics and gives the router a
        # gradient path through the selected outputs.
        sel_scores = torch.gather(scores, 1, top_idx)       # (B, k)
        out = gathered + torch.sigmoid(sel_scores).unsqueeze(-1).to(out.dtype) * (out - gathered)
        return x.scatter(1, top_idx.unsqueeze(-1).expand(-1, -1, D), out)

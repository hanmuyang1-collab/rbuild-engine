"""Fine-grained MoE with shared expert(s), and the dense FFN fallback."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseFFN(nn.Module):
    """Gated SiLU FFN."""

    def __init__(self, d_model: int, ffn_dim: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, ffn_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class FineGrainedMoE(nn.Module):
    """
    Fine-grained routed experts (each small) + always-on shared expert(s).
    Token-choice top-k routing with a softmax gate; a lightweight load-
    balancing auxiliary loss is exposed for the trainer.
    """

    def __init__(self, d_model: int, n_experts: int, top_k: int,
                 expert_dim: int, n_shared: int = 1, shared_dim: Optional[int] = None):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList(DenseFFN(d_model, expert_dim) for _ in range(n_experts))
        # shared expert(s) stay dense-sized (DeepSeek-style): they are the
        # always-on capacity, sized from the model width, not from the
        # fine-grained expert width.
        per_shared = shared_dim or max(expert_dim, int(d_model * 4 / max(1, n_shared)))
        self.shared = nn.ModuleList(DenseFFN(d_model, per_shared) for _ in range(n_shared))
        self.last_aux_loss = torch.zeros(())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        flat = x.reshape(-1, D)                              # (N, D)
        logits = self.router(flat)                           # (N, E)
        probs = F.softmax(logits.float(), dim=-1)
        top_p, top_i = probs.topk(self.top_k, dim=-1)        # (N, k)
        top_p = top_p / top_p.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        # v4.3: sort-based grouped dispatch. v4.2 looped over every expert
        # with a boolean mask, `mask.any()` and `nonzero()` — 2 device->host
        # syncs per expert per layer. Here tokens are sorted by expert once,
        # one bincount is read back, and each expert runs on a contiguous
        # slice. Outputs are combined with a single index_add_.
        N, K = top_i.shape
        flat_e = top_i.reshape(-1)                           # (N*k,)
        order = torch.argsort(flat_e, stable=True)
        token_of = torch.div(order, K, rounding_mode="floor")
        weight_of = top_p.reshape(-1)[order].to(flat.dtype)
        counts_t = torch.bincount(flat_e, minlength=self.n_experts)
        counts = counts_t.tolist()                           # the one sync
        routed = flat.index_select(0, token_of)              # (N*k, D) sorted
        pieces, start = [], 0
        for e, c in enumerate(counts):
            if c:
                pieces.append(self.experts[e](routed[start:start + c]))
            start += c
        out = torch.zeros_like(flat)
        if pieces:
            expert_out = torch.cat(pieces, 0) * weight_of.unsqueeze(-1)
            out.index_add_(0, token_of, expert_out.to(out.dtype))

        # load-balancing aux loss (Switch-style), kept in fp32
        if self.training:
            mean_prob = probs.mean(0)
            mean_load = counts_t.float()
            mean_load = mean_load / mean_load.sum().clamp_min(1.0)
            self.last_aux_loss = self.n_experts * (mean_prob * mean_load).sum()
        else:
            self.last_aux_loss = torch.zeros((), device=x.device)

        shared_out = sum(s(flat) for s in self.shared) / max(1, len(self.shared))
        return (out + shared_out).reshape(B, T, D)


def build_ffn(kind: str, d_model: int, *, n_experts: int = 8, top_k: int = 2,
              expert_dim: Optional[int] = None, n_shared: int = 1,
              dense_dim: Optional[int] = None, ffn_mult: float = 4.0) -> nn.Module:
    if kind == "moe":
        exp_dim = expert_dim or max(8, int(d_model * ffn_mult / n_experts))
        return FineGrainedMoE(d_model, n_experts, top_k, exp_dim, n_shared)
    return DenseFFN(d_model, dense_dim or int(d_model * ffn_mult))

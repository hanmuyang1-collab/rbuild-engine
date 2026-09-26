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

        out = torch.zeros_like(flat)
        for e in range(self.n_experts):
            mask = top_i == e                                # (N, k)
            if not mask.any():
                continue
            token_idx, slot = mask.nonzero(as_tuple=True)
            expert_out = self.experts[e](flat[token_idx])
            out.index_add_(0, token_idx, expert_out * top_p[token_idx, slot].unsqueeze(-1).to(flat.dtype))

        # load-balancing aux loss (Switch-style)
        if self.training:
            mean_prob = probs.mean(0)
            mean_load = torch.zeros(self.n_experts, device=x.device)
            mean_load.scatter_add_(0, top_i.reshape(-1),
                                   torch.ones_like(top_i, dtype=torch.float32).reshape(-1))
            mean_load = mean_load / mean_load.sum().clamp_min(1.0)
            self.last_aux_loss = (self.n_experts * (mean_prob * mean_load).sum()).to(x.dtype)
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

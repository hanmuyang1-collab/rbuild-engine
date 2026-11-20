"""
Muon optimizer (Newton-Schulz orthogonalized momentum for 2-D hidden
matrices, AdamW for embeddings / norms / 1-D params) and the WSD
(warmup-stable-decay) learning-rate schedule.
"""

from __future__ import annotations

import math
from typing import Iterable, List

import torch
from torch import nn


@torch.no_grad()
def _zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5,
                                 bf16: bool = False) -> torch.Tensor:
    """Newton-Schulz iteration to compute the orthogonalization of G.
    v4.3: bf16=True runs the iteration in bf16 (tensor-core speed, the
    reference Muon setting); fp32 stays the default off-GPU."""
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() if bf16 else G.float()
    X = X / (X.norm() + 1e-7)
    transposed = G.shape[0] > G.shape[1]
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon for 2-D hidden weights. All values user-tunable."""

    def __init__(self, params: Iterable, lr: float = 3e-3, momentum: float = 0.95,
                 ns_steps: int = 5, weight_decay: float = 0.0, bf16: bool = True):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps,
                        weight_decay=weight_decay, bf16=bf16)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr, mom = group["lr"], group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(mom).add_(g)
                g = g.add(buf, alpha=mom)          # nesterov
                # v4.3: Muon needs the WHOLE matrix. Under FSDP2 the grad is a
                # sharded DTensor; orthogonalising a shard is mathematically
                # wrong, so gather, orthogonalise, and re-shard the update.
                dt_mesh = getattr(g, "device_mesh", None)
                placements = getattr(g, "placements", None)
                if dt_mesh is not None and hasattr(g, "full_tensor"):
                    g = g.full_tensor()
                g2 = g.reshape(g.shape[0], -1) if g.ndim > 2 else g
                og = _zeropower_via_newtonschulz5(
                    g2, steps=group["ns_steps"],
                    bf16=bool(group.get("bf16", True)) and g2.is_cuda)
                og = og.reshape_as(g).to(p.dtype)
                scale = max(1.0, g.shape[0] / g.reshape(g.shape[0], -1).shape[1]) ** 0.5
                if dt_mesh is not None:
                    from torch.distributed.tensor import distribute_tensor
                    og = distribute_tensor(og, dt_mesh, placements)
                if group["weight_decay"] > 0:
                    p.mul_(1 - lr * group["weight_decay"])
                p.add_(og, alpha=-lr * scale)
        return loss


def build_optimizer(model: nn.Module, cfg) -> List[torch.optim.Optimizer]:
    """
    Split parameters: Muon gets 2-D hidden matrices; AdamW gets embeddings,
    norms, routers, gates and anything 1-D. Returns a list — the trainer
    steps both.
    """
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and "embed" not in name and "lm_head" not in name \
                and "router" not in name and "gate" not in name \
                and "fact_" not in name and "mem_key_proj" not in name:
            muon_params.append(p)
        else:
            adamw_params.append(p)
    opts = []
    tc = cfg.train
    on_cuda = any(p.is_cuda for p in muon_params + adamw_params)

    def adamw(params, lr, **kw):
        # v4.3: 8-bit states (bitsandbytes) cut Adam from 8 to 2 bytes/param;
        # otherwise the fused CUDA kernel (one launch for all params).
        if getattr(tc, "optimizer_8bit", False):
            try:
                import bitsandbytes as bnb
                return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=tc.weight_decay, **kw)
            except ImportError:
                pass
        extra = {"fused": True} if on_cuda else {"foreach": True}
        try:
            return torch.optim.AdamW(params, lr=lr, weight_decay=tc.weight_decay, **kw, **extra)
        except (RuntimeError, TypeError):
            return torch.optim.AdamW(params, lr=lr, weight_decay=tc.weight_decay, **kw)

    if muon_params and tc.optimizer == "muon":
        opts.append(Muon(muon_params, lr=tc.lr,
                         momentum=tc.muon_momentum,
                         ns_steps=tc.muon_ns_steps,
                         weight_decay=tc.weight_decay,
                         bf16=getattr(tc, "muon_bf16", True)))
    elif muon_params:
        opts.append(adamw(muon_params, tc.adamw_lr))
    if adamw_params:
        opts.append(adamw(adamw_params, tc.adamw_lr, betas=(0.9, 0.95)))
    if not opts:
        raise ValueError("No trainable parameters were supplied to the optimizer")
    return opts


class WSDScheduler:
    """
    Warmup-Stable-Decay. `cooldown_frac` of the run is the decay tail.
    All user-modifiable through TrainConfig.
    """

    def __init__(self, optimizers: List[torch.optim.Optimizer], base_lrs: List[List[float]],
                 warmup_steps: int, max_steps: int, cooldown_frac: float = 0.4):
        self.opts = optimizers
        self.base_lrs = base_lrs
        self.warmup = max(1, warmup_steps)
        self.max_steps = max(1, max_steps)
        self.cooldown_frac = min(max(cooldown_frac, 0.0), 1.0)
        self.decay_start = int(self.max_steps * (1 - self.cooldown_frac))

    def factor(self, step: int) -> float:
        if step < self.warmup:
            return (step + 1) / self.warmup
        if step < self.decay_start:
            return 1.0
        t = (step - self.decay_start) / max(1, self.max_steps - self.decay_start)
        return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))   # cosine tail

    def set(self, step: int) -> float:
        f = self.factor(step)
        for opt, lrs in zip(self.opts, self.base_lrs):
            for group, base in zip(opt.param_groups, lrs):
                group["lr"] = base * f
        return f

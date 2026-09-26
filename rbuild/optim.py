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
def _zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration to compute the orthogonalization of G."""
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
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
                 ns_steps: int = 5, weight_decay: float = 0.0):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps, weight_decay=weight_decay)
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
                g2 = g.reshape(g.shape[0], -1) if g.ndim > 2 else g
                og = _zeropower_via_newtonschulz5(g2, steps=group["ns_steps"])
                og = og.reshape_as(g)
                scale = max(1.0, g.shape[0] / g.reshape(g.shape[0], -1).shape[1]) ** 0.5
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
    if muon_params and cfg.train.optimizer == "muon":
        opts.append(Muon(muon_params, lr=cfg.train.lr,
                         momentum=cfg.train.muon_momentum,
                         ns_steps=cfg.train.muon_ns_steps,
                         weight_decay=cfg.train.weight_decay))
    elif muon_params:
        opts.append(torch.optim.AdamW(muon_params, lr=cfg.train.adamw_lr,
                                      weight_decay=cfg.train.weight_decay))
    opts.append(torch.optim.AdamW(adamw_params, lr=cfg.train.adamw_lr,
                                  weight_decay=cfg.train.weight_decay,
                                  betas=(0.9, 0.95)))
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

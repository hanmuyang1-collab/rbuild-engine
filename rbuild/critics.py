"""
R-Build v3 — critic experts and ACT-style adaptive halting.

The v3 core idea: the model no longer trusts its own working layers
blindly. Two places get critics:

1. Extraction loop (the cache-loop line). The loop now runs *adaptively*:
   after each iteration a panel of X parallel critic experts scores every
   token's hidden state. A token halts once its cumulative halting score
   crosses 1 (ACT-style); the whole loop early-exits once at least Y
   critics report "satisfied" on average. An ACT ponder loss
   (loops_taken + remainder) regularizes the compute spent, so the model
   learns to extract only as much as each token needs.

2. Generative layers (the parallel bundle stages). Each stage carries X
   parallel critic experts with *more capacity than the working experts*
   (critic_hidden = expert_ffn_dim * critic_capacity_mult). They score the
   stage bundle and are the verifiers used by the noting experts —
   self-training data only becomes real once the critics approve it.

Everything is user-modifiable through CriticConfig; `enabled=False`
removes every critic parameter and makes v3 bit-identical to v2.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CriticExpert(nn.Module):
    """
    One critic: scores how *complete* a hidden state is.

    Returns a satisfaction probability per token in [0, 1]. Critics are
    intentionally wider than the working experts they judge — verification
    is easier than generation, but it still needs capacity headroom to be
    a trustworthy gate.
    """

    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, hidden, bias=False)
        self.score = nn.Linear(hidden, 1, bias=True)
        nn.init.zeros_(self.score.bias)  # start neutral: p = 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> satisfaction (B, T) in [0, 1]."""
        h = F.silu(self.w1(x))
        h = F.silu(self.w2(h))
        return torch.sigmoid(self.score(h)).squeeze(-1)


class CriticPanel(nn.Module):
    """
    X parallel critic experts over the same hidden state.

    forward() returns:
      satisfaction : (B, T) mean satisfaction across critics
      n_satisfied  : (B, T) how many critics are >= threshold (the "Y" vote)
    """

    def __init__(self, d_model: int, hidden: int, n_critics: int,
                 threshold: float = 0.5):
        super().__init__()
        self.n_critics = n_critics
        self.threshold = threshold
        self.critics = nn.ModuleList(CriticExpert(d_model, hidden)
                                     for _ in range(n_critics))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        votes = torch.stack([c(x) for c in self.critics], dim=-1)  # (B, T, X)
        satisfaction = votes.mean(-1)
        n_satisfied = (votes >= self.threshold).sum(-1)            # (B, T)
        return satisfaction, n_satisfied


class ACTHalting:
    """
    ACT-style adaptive halting state for the extraction loop.

    Per token, the critic satisfaction at loop iteration l acts as the
    halting probability mass contributed that iteration. A token stops
    updating once its cumulative mass reaches 1 - eps; the residual mass
    (remainder) is counted in the ponder cost so the total mass per token
    is exactly 1 + remainder, and minimizing the ponder cost teaches the
    critics to fire early.

    The loop as a whole exits early once, on average over still-running
    tokens, at least `y_critics` critics are satisfied — the "extraction
    runs until Y critics are satisfied" rule.

    Usage (inside CacheLoopLine):
        halt = ACTHalting(cfg.critic, batch, seq, device)
        for loop in range(max_loops):
            x = run_one_loop(x)
            if halt.step(x, critic_panel): break
        ponder_cost = halt.ponder_cost()
    """

    def __init__(self, cfg, batch: int, seq: int, device,
                 max_loops: Optional[int] = None, y_critics: Optional[int] = None):
        self.y_critics = y_critics if y_critics is not None else cfg.y_critics
        self.max_loops = max_loops if max_loops is not None else cfg.max_loops
        self.eps = cfg.halt_eps
        self.min_loops = min(cfg.min_loops, self.max_loops)
        self.cumulative = torch.zeros(batch, seq, device=device)   # halting mass
        self.n_loops = torch.zeros(batch, seq, device=device)      # iterations used
        self.remainders = torch.zeros(batch, seq, device=device)
        self.running = torch.ones(batch, seq, dtype=torch.bool, device=device)
        self.iterations = 0

    def step(self, x: torch.Tensor, panel: CriticPanel) -> bool:
        """
        Register one finished loop iteration over hidden state x.
        Returns True when the whole loop should stop.
        """
        satisfaction, n_satisfied = panel(x)                       # (B, T)
        self.iterations += 1
        last = self.iterations >= self.max_loops

        # tokens still running gain this iteration's halting mass
        p = satisfaction.clamp(0.0, 1.0)
        p = torch.where(self.running, p, torch.zeros_like(p))

        new_cum = self.cumulative + p
        halting = (new_cum >= 1.0 - self.eps) | last
        halting = halting & self.running

        # remainder closes the ACT mass to exactly 1 for newly halted tokens
        rem = (1.0 - self.cumulative).clamp_min(0.0)
        self.remainders = torch.where(halting, rem, self.remainders)
        self.n_loops = torch.where(halting | self.running,
                                   torch.full_like(self.n_loops, float(self.iterations)),
                                   self.n_loops)
        self.cumulative = torch.where(self.running, new_cum, self.cumulative)
        self.running = self.running & ~halting

        if self.iterations < self.min_loops:
            return False
        if last:
            self.running.zero_()
            return True
        if not self.running.any():
            return True
        # global early exit: running tokens already satisfy >= Y critics
        votes = n_satisfied[self.running].float()
        return bool(votes.mean().item() >= self.y_critics)

    def token_weights(self) -> torch.Tensor:
        """ACT update weights: per-token mass used to scale state updates."""
        w = self.cumulative.clamp(max=1.0) + self.remainders
        return w.clamp_min(1e-3)

    def ponder_cost(self) -> torch.Tensor:
        """Mean (loops_taken + remainder) — the ACT halting loss term."""
        return (self.n_loops + self.remainders).mean()

    def stats(self) -> dict:
        return {
            "mean_loops": float(self.n_loops.mean().item()),
            "max_loop": int(self.n_loops.max().item()) if self.iterations else 0,
            "ponder_cost": float(self.ponder_cost().item()),
        }

"""
R-Build v3 — noting experts + critic-verified non-separate self-training.

v2 could absorb facts without training (gradient-free delta-rule writes).
v3 closes the loop: the model *watches its own forward passes*, takes
notes, and — once the critic experts verify a note — learns from it
*while running*. No separate training phase, no separate verification
phase: extraction, noting, verification and learning all ride the same
forward, so this is "non-separate" self-training.

Pipeline per forward (when noting.enabled):
    hidden states -> NotingExperts      : candidate notes (key, value, conf)
                  -> CriticPanel verify : only notes approved by >= Y critics
                                          with mean satisfaction >= threshold
                                          are kept
                  -> fast-weight write  : verified notes enter the delta-rule
                                          memory immediately (gradient-free,
                                          ~zero extra RAM — this is the
                                          "learn while running" path)
                  -> VerifiedNoteBuffer : verified notes also queue on CPU
                                          (low-RAM) for real gradient steps
                                          via Trainer.self_train_step()

The gradient path replays verified notes through the memory read machinery
with a reconstruction loss, so slow weights consolidate what the fast
weights already absorbed — parallel running and learning, least RAM:
notes live on CPU in fp16 until the moment they are trained on.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class NotingExpert(nn.Module):
    """
    One note-taker: reads the hidden stream and writes a candidate fact.

    Produces a (key, value) pair in the fast-weight memory's format plus a
    confidence. Notes are just proposed writes — they only stick if the
    critics approve.
    """

    def __init__(self, d_model: int, key_dim: int, value_dim: int, hidden: int):
        super().__init__()
        self.pre = nn.Linear(d_model, hidden, bias=False)
        self.key_proj = nn.Linear(hidden, key_dim, bias=False)
        self.value_proj = nn.Linear(hidden, value_dim, bias=False)
        self.conf = nn.Linear(hidden, 1, bias=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: (B, T, D) -> keys (B,T,kd), values (B,T,vd), confidence (B,T)."""
        h = F.silu(self.pre(x))
        return self.key_proj(h), self.value_proj(h), torch.sigmoid(self.conf(h)).squeeze(-1)


class NotingExperts(nn.Module):
    """The panel of note-taking experts; notes are pooled across experts."""

    def __init__(self, d_model: int, key_dim: int, value_dim: int,
                 n_experts: int, hidden: int):
        super().__init__()
        self.experts = nn.ModuleList(
            NotingExpert(d_model, key_dim, value_dim, hidden) for _ in range(n_experts))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        keys, values, confs = [], [], []
        for e in self.experts:
            k, v, c = e(x)
            keys.append(k); values.append(v); confs.append(c)
        # confidence-weighted pool across the note-takers
        w = F.softmax(torch.stack(confs, -1), dim=-1)              # (B, T, E)
        key = (torch.stack(keys, -1) * w.unsqueeze(2)).sum(-1)     # (B, T, kd)
        val = (torch.stack(values, -1) * w.unsqueeze(2)).sum(-1)   # (B, T, vd)
        conf = torch.stack(confs, -1).mean(-1)                     # (B, T)
        return key, val, conf


class VerifiedNoteBuffer:
    """
    Low-RAM queue of critic-verified notes awaiting a real gradient step.

    Each entry keeps the note (key, value, score) plus the hidden state it
    was taken from — everything on CPU in fp16 (half footprint), moved to
    the GPU only for the consolidation step itself. This is what makes
    non-separate self-training cheap: observation is continuous, GPU work
    is batched and rare.
    """

    def __init__(self, capacity: int = 4096):
        self.capacity = capacity
        self.hidden: List[torch.Tensor] = []
        self.keys: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.scores: List[torch.Tensor] = []

    def __len__(self) -> int:
        return sum(k.shape[0] for k in self.keys)

    @torch.no_grad()
    def add(self, hidden: torch.Tensor, keys: torch.Tensor,
            values: torch.Tensor, scores: torch.Tensor) -> None:
        self.hidden.append(hidden.detach().to("cpu", torch.float16))
        self.keys.append(keys.detach().to("cpu", torch.float16))
        self.values.append(values.detach().to("cpu", torch.float16))
        self.scores.append(scores.detach().to("cpu", torch.float16))
        while len(self) > self.capacity and self.keys:
            overflow = len(self) - self.capacity
            head = self.keys[0].shape[0]
            if head <= overflow:
                for lst in (self.hidden, self.keys, self.values, self.scores):
                    lst.pop(0)
            else:
                for lst in (self.hidden, self.keys, self.values, self.scores):
                    lst[0] = lst[0][overflow:]

    def sample(self, n: int):
        """Pop up to n notes (highest-scoring first). None if empty."""
        if not self.keys:
            return None
        h = torch.cat(self.hidden); k = torch.cat(self.keys)
        v = torch.cat(self.values); s = torch.cat(self.scores)
        n = min(n, k.shape[0])
        idx = s.topk(n).indices
        out = (h[idx].float(), k[idx].float(), v[idx].float(), s[idx].float())
        keep = torch.ones(k.shape[0], dtype=torch.bool)
        keep[idx] = False
        self.hidden = [h[keep]] if keep.any() else []
        self.keys = [k[keep]] if keep.any() else []
        self.values = [v[keep]] if keep.any() else []
        self.scores = [s[keep]] if keep.any() else []
        return out

    def clear(self) -> None:
        self.hidden.clear(); self.keys.clear()
        self.values.clear(); self.scores.clear()


class SelfLearner:
    """
    Orchestrates non-separate self-training for one RBuildModel.

    Wired into the model's forward (see RBuildModel._self_observe): notes
    are taken from the final hidden states, verified by the generative
    stages' critics, and verified notes are (a) written into fast-weight
    memory gradient-free and (b) queued for Trainer.self_train_step().
    """

    def __init__(self, model):
        self.model = model
        cfg = model.cfg.noting
        self.buffer = VerifiedNoteBuffer(cfg.buffer_capacity)
        self.n_verified = 0
        self.n_rejected = 0

    @torch.no_grad()
    def observe(self, hidden: torch.Tensor) -> int:
        """
        hidden: (B, T, D) final hidden states of a forward pass.
        Returns how many notes survived critic verification this pass.
        """
        model, cfg = self.model, self.model.cfg.noting
        keys, values, conf = model.noting_experts(hidden)           # (B,T,kd/vd),(B,T)
        satisfaction, n_satisfied = model.critics_verify(hidden)    # critics' verdict

        keep = (n_satisfied >= cfg.verify_y_critics) \
             & (satisfaction >= cfg.verify_threshold) \
             & (conf >= cfg.min_confidence)                         # (B, T)
        n_keep = int(keep.sum())
        self.n_rejected += int((~keep).sum())
        if n_keep == 0:
            return 0

        k = keys[keep]; v = values[keep]; s = satisfaction[keep] * conf[keep]
        # (a) gradient-free: fast-weight memory absorbs verified notes now
        if cfg.write_to_memory and model.memory is not None:
            model.memory.write(k, v, lr=cfg.memory_write_lr)
        # (b) queued: slow-weight consolidation later (low-RAM CPU buffer)
        self.buffer.add(hidden[keep], k, v, s)
        self.n_verified += n_keep
        return n_keep

    def stats(self) -> dict:
        total = self.n_verified + self.n_rejected
        return {
            "notes_verified": self.n_verified,
            "notes_rejected": self.n_rejected,
            "accept_rate": (self.n_verified / total) if total else 0.0,
            "buffered": len(self.buffer),
        }

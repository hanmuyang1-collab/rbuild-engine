"""
Fast-weight memory: a delta-rule key->value matrix.

Writes are *gradient-free* — `write_fact(text_or_keys, values)` updates the
matrix in place with the delta rule, so the model can absorb facts without
any training step. The matrix is a registered buffer, so it is persisted
with every checkpoint automatically.

Delta rule:
    M <- decay * M + lr * (v - M @ k_norm) ⊗ k_norm        (write)
    read(k) = M @ k_norm                                    (read)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FastWeightMemory(nn.Module):
    def __init__(self, key_dim: int, value_dim: int, write_lr: float = 1.0,
                 decay: float = 0.999, max_facts: int = 65536):
        super().__init__()
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.write_lr = write_lr
        self.decay = decay
        self.max_facts = max_facts
        # the fast-weight matrix itself — a buffer, saved with checkpoints
        self.register_buffer("matrix", torch.zeros(key_dim, value_dim))
        self.register_buffer("n_writes", torch.zeros((), dtype=torch.long))
        self._fact_log: list = []   # human-readable record of what was written

    # ------------------------------------------------------------------ #
    def _norm(self, k: torch.Tensor) -> torch.Tensor:
        return F.normalize(k.float(), dim=-1)

    # ------------------------------------------------------------------ #
    def read(self, keys: torch.Tensor) -> torch.Tensor:
        """keys: (..., key_dim) -> retrieved values (..., value_dim)."""
        return self._norm(keys) @ self.matrix.to(keys.dtype)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def write(self, keys: torch.Tensor, values: torch.Tensor,
              lr: Optional[float] = None) -> None:
        """
        Delta-rule write. keys: (N, key_dim), values: (N, value_dim).
        No gradients involved — this is the "learn facts without training" path.
        """
        lr = self.write_lr if lr is None else lr
        k = self._norm(keys)                                # (N, kd)
        v = values.float()                                  # (N, vd)
        m = self.matrix.float()
        pred = k @ m                                        # what we currently recall
        delta = v - pred                                    # (N, vd)
        # batch delta-rule update
        m = m * (self.decay ** k.shape[0]) + lr * (k.T @ delta) / max(1, k.shape[0])
        self.matrix.copy_(m.to(self.matrix.dtype))
        self.n_writes += k.shape[0]

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self.matrix.zero_()
        self.n_writes.zero_()
        self._fact_log.clear()

    def extra_repr(self) -> str:
        return (f"key_dim={self.key_dim}, value_dim={self.value_dim}, "
                f"writes={int(self.n_writes)}, decay={self.decay}")

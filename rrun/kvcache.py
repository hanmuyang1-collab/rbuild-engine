"""KV cache manager.

R-Run's swap promise: the old model is wiped completely, but the NEW model
always comes up with its FULL KV cache pre-allocated — never a crippled,
truncated cache. Swap speed comes from wiping + reallocating, not from
shrinking the cache.

Note: KV cache is architecture- and weights-specific, so it can never be
carried across different models. "Keeping full KV cache" therefore means:
every swap ends with the new model holding a full-capacity, freshly
initialized cache sized for its max context.
"""
from __future__ import annotations

import gc
import sys
import threading
from dataclasses import dataclass, field


def _bytes_per_token_kv(n_layers: int, n_kv_heads: int, head_dim: int,
                        dtype_bytes: int = 2) -> int:
    # K and V for every layer
    return 2 * n_layers * n_kv_heads * head_dim * dtype_bytes


@dataclass
class KVCacheSpec:
    model_id: str
    max_context: int
    n_layers: int
    n_kv_heads: int
    head_dim: int
    dtype_bytes: int = 2
    max_seqs: int = 256                     # max concurrent sequences

    @property
    def bytes_per_token(self) -> int:
        return _bytes_per_token_kv(self.n_layers, self.n_kv_heads,
                                   self.head_dim, self.dtype_bytes)

    @property
    def capacity_tokens(self) -> int:
        return self.max_context * self.max_seqs

    @property
    def total_gb(self) -> float:
        return self.bytes_per_token * self.capacity_tokens / (1024 ** 3)


class KVCacheManager:
    """Tracks the single resident KV cache. Full-wipe semantics on swap."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._spec: KVCacheSpec | None = None
        self._allocated_tokens: int = 0
        self.wipe_count: int = 0

    def allocate_full(self, spec: KVCacheSpec) -> KVCacheSpec:
        """Pre-allocate the entire cache for a freshly loaded model."""
        with self._lock:
            if self._spec is not None:
                raise RuntimeError(
                    "KV cache still held by "
                    f"{self._spec.model_id!r}; wipe() before allocate_full()."
                )
            self._spec = spec
            self._allocated_tokens = spec.capacity_tokens
            return spec

    def wipe(self) -> None:
        """Drop every cache block and force host/GPU memory release."""
        with self._lock:
            self._spec = None
            self._allocated_tokens = 0
            self.wipe_count += 1
        gc.collect()
        # Only touch CUDA if torch is already loaded by the active backend —
        # importing torch here would add seconds to an otherwise instant wipe.
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @property
    def resident(self) -> bool:
        return self._spec is not None

    def status(self) -> dict:
        with self._lock:
            if self._spec is None:
                return {"resident": False, "allocated_tokens": 0,
                        "wipe_count": self.wipe_count}
            return {
                "resident": True,
                "model_id": self._spec.model_id,
                "allocated_tokens": self._allocated_tokens,
                "capacity_tokens": self._spec.capacity_tokens,
                "gb": round(self._spec.total_gb, 3),
                "full": self._allocated_tokens == self._spec.capacity_tokens,
                "wipe_count": self.wipe_count,
            }

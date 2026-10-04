"""Mock backend — smoke tests and CPU-only dev boxes. No weights touched."""
from __future__ import annotations

import hashlib
import time

from .base import Backend, LoadReport


class MockBackend(Backend):
    name = "mock"

    def __init__(self, simulated_vram_gb: float = 4.0,
                 simulated_load_s: float = 0.05,
                 max_context: int = 32768) -> None:
        self._vram = simulated_vram_gb
        self._load_s = simulated_load_s
        self._max_context = max_context
        self._model_id: str | None = None

    def load(self, model_id: str, **kw) -> LoadReport:
        time.sleep(self._load_s)
        self._model_id = model_id
        # Deterministic fake geometry derived from the model id
        h = int(hashlib.sha256(model_id.encode()).hexdigest()[:8], 16)
        n_layers, n_kv_heads, head_dim = 28 + h % 8, 4 + h % 4, 128
        kv_gb = (2 * n_layers * n_kv_heads * head_dim * 2
                 * self._max_context * 64) / (1024 ** 3)
        return LoadReport(model_id=model_id, load_seconds=self._load_s,
                          vram_gb=self._vram, kv_cache_gb=round(kv_gb, 3),
                          max_context=self._max_context, backend=self.name,
                          extra={"n_layers": n_layers, "n_kv_heads": n_kv_heads,
                                 "head_dim": head_dim})

    def unload(self) -> None:
        self._model_id = None

    def generate(self, prompt: str, max_tokens: int = 128,
                 temperature: float = 0.7, **kw) -> str:
        return (f"[{self._model_id}] mock reply to "
                f"{prompt[-64:]!r} (max_tokens={max_tokens})")

    @property
    def loaded(self) -> bool:
        return self._model_id is not None

"""RRunEngine — resident-model manager with single-call hot swap.

Swap sequence (the heart of R-Run):
    1. take the swap lock (in-flight requests finish or fail fast)
    2. wipe the old KV cache + unload old weights  -> VRAM fully released
    3. load the new model                          -> weights in
    4. allocate the FULL KV cache for the new model
    5. release the lock                            -> server never restarted

The HTTP server keeps listening on the same port the whole time, so clients
never reconnect. `rrun swap <model>` is the one command that drives this.
"""
from __future__ import annotations

import threading
import time
from dataclasses import asdict

from .backends.base import Backend, LoadReport
from .kvcache import KVCacheManager, KVCacheSpec


class ModelNotLoadedError(RuntimeError):
    pass


class SwapInProgressError(RuntimeError):
    pass


class RRunEngine:
    def __init__(self, backend: Backend, max_context: int = 32768,
                 max_seqs: int = 256) -> None:
        self.backend = backend
        self.kv = KVCacheManager()
        self.max_context = max_context
        self.max_seqs = max_seqs
        self._swap_lock = threading.Lock()
        self._current: LoadReport | None = None
        self.history: list[dict] = []

    # ------------------------------------------------------------------ load
    def _load_and_cache(self, model_id: str, **load_kw) -> LoadReport:
        report = self.backend.load(model_id, **load_kw)
        spec = KVCacheSpec(
            model_id=model_id,
            max_context=report.max_context,
            n_layers=report.extra.get("n_layers", 32),
            n_kv_heads=report.extra.get("n_kv_heads", 8),
            head_dim=report.extra.get("head_dim", 128),
            max_seqs=self.max_seqs,
        )
        self.kv.allocate_full(spec)
        return report

    def serve(self, model_id: str, **load_kw) -> LoadReport:
        """Cold start: load a model when nothing is resident."""
        with self._swap_lock:
            if self.backend.loaded:
                raise RuntimeError(
                    f"{self._current.model_id!r} already resident; "
                    "use swap() instead of serve().")
            self._current = self._load_and_cache(model_id, **load_kw)
            self._record("serve", self._current)
            return self._current

    # ------------------------------------------------------------------ swap
    def swap(self, model_id: str, **load_kw) -> LoadReport:
        """THE one-command swap: wipe resident model, host a new one.

        Full KV cache is guaranteed for the incoming model — the cache is
        rebuilt at full capacity on every swap, never shrunk for speed.
        """
        if not self._swap_lock.acquire(blocking=False):
            raise SwapInProgressError("a swap is already in progress")
        try:
            old = self._current.model_id if self._current else None
            t0 = time.perf_counter()

            # 1+2. wipe: KV cache first, then weights — nothing of the old
            #      model survives.
            self.kv.wipe()
            self.backend.unload()
            self._current = None

            # 3+4. load new weights + allocate FULL KV cache.
            self._current = self._load_and_cache(model_id, **load_kw)

            dt = time.perf_counter() - t0
            self._record("swap", self._current, old=old, swap_seconds=dt)
            return self._current
        except Exception:
            # Never leave the engine half-wiped without a resident model note
            self._record("swap_failed", None, old=old)
            raise
        finally:
            self._swap_lock.release()

    # ------------------------------------------------------------- inference
    def generate(self, prompt: str, **kw) -> str:
        self._require_model()
        return self.backend.generate(prompt, **kw)

    def chat(self, messages: list[dict], **kw) -> str:
        self._require_model()
        return self.backend.chat_generate(messages, **kw)

    # ---------------------------------------------------------------- status
    def status(self) -> dict:
        return {
            "model": asdict(self._current) if self._current else None,
            "kv_cache": self.kv.status(),
            "backend": self.backend.name,
            "swaps": len([h for h in self.history if h["event"] == "swap"]),
        }

    def _require_model(self) -> None:
        if not self.backend.loaded:
            raise ModelNotLoadedError(
                "no model resident — run `rrun serve <model>` first")

    def _record(self, event: str, report: LoadReport | None, **extra) -> None:
        entry = {"event": event, "ts": time.time(), **extra}
        if report is not None:
            entry["model_id"] = report.model_id
            entry["kv_cache_gb"] = report.kv_cache_gb
        self.history.append(entry)

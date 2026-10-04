"""vLLM backend — the fast path. Full KV cache via gpu_memory_utilization."""
from __future__ import annotations

import gc
import time

from .base import Backend, LoadReport


class VLLMBackend(Backend):
    name = "vllm"

    def __init__(self, gpu_memory_utilization: float = 0.90,
                 max_model_len: int | None = None,
                 quantization: str | None = None,
                 dtype: str = "auto",
                 tensor_parallel_size: int = 1,
                 trust_remote_code: bool = True) -> None:
        self._kw = dict(
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            quantization=quantization,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
        )
        self._llm = None
        self._model_id: str | None = None

    def load(self, model_id: str, **kw) -> LoadReport:
        from vllm import LLM  # imported lazily; only needed on GPU hosts

        args = {k: v for k, v in self._kw.items() if v is not None}
        args.update(kw)
        t0 = time.perf_counter()
        self._llm = LLM(model=model_id, **args)
        self._model_id = model_id
        dt = time.perf_counter() - t0

        cfg = self._llm.llm_engine.model_config
        cache_cfg = self._llm.llm_engine.cache_config
        max_len = args.get("max_model_len") or cfg.max_model_len
        try:
            import torch
            vram = torch.cuda.memory_allocated() / (1024 ** 3)
        except Exception:
            vram = 0.0
        return LoadReport(
            model_id=model_id, load_seconds=round(dt, 2),
            vram_gb=round(vram, 2),
            kv_cache_gb=round(
                getattr(cache_cfg, "num_gpu_blocks", 0)
                * getattr(cache_cfg, "block_size", 16)
                * self._bytes_per_block_token(cfg) / (1024 ** 3), 2),
            max_context=max_len, backend=self.name,
        )

    @staticmethod
    def _bytes_per_block_token(cfg) -> float:
        try:
            return (2 * cfg.get_num_layers() * cfg.get_num_kv_heads()
                    * cfg.get_head_size() * 2)
        except Exception:
            return 0.0

    def unload(self) -> None:
        llm, self._llm = self._llm, None
        self._model_id = None
        if llm is not None:
            try:
                # vLLM >= 0.9: explicit engine shutdown frees KV blocks
                llm.llm_engine.engine_core.shutdown()  # type: ignore[attr-defined]
            except Exception:
                pass
            del llm
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass

    def generate(self, prompt: str, max_tokens: int = 128,
                 temperature: float = 0.7, **kw) -> str:
        from vllm import SamplingParams
        sp = SamplingParams(max_tokens=max_tokens, temperature=temperature)
        out = self._llm.generate([prompt], sp)
        return out[0].outputs[0].text

    @property
    def loaded(self) -> bool:
        return self._llm is not None

"""HF transformers backend — works anywhere, including CPU-only boxes."""
from __future__ import annotations

import gc
import time

from .base import Backend, LoadReport


class HFBackend(Backend):
    name = "hf"

    def __init__(self, device_map: str = "auto",
                 torch_dtype: str = "auto",
                 max_context: int = 4096,
                 trust_remote_code: bool = True) -> None:
        self._device_map = device_map
        self._torch_dtype = torch_dtype
        self._max_context = max_context
        self._trust_remote_code = trust_remote_code
        self._model = None
        self._tok = None
        self._model_id: str | None = None

    def load(self, model_id: str, **kw) -> LoadReport:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        t0 = time.perf_counter()
        self._tok = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=self._trust_remote_code)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map=self._device_map,
            torch_dtype=self._torch_dtype,
            trust_remote_code=self._trust_remote_code, **kw)
        self._model_id = model_id
        dt = time.perf_counter() - t0

        cfg = self._model.config
        n_layers = getattr(cfg, "num_hidden_layers", 0)
        n_kv = getattr(cfg, "num_key_value_heads",
                       getattr(cfg, "num_attention_heads", 0))
        head_dim = getattr(cfg, "head_dim",
                           getattr(cfg, "hidden_size", 0) //
                           max(getattr(cfg, "num_attention_heads", 1), 1))
        max_ctx = min(getattr(cfg, "max_position_embeddings", self._max_context),
                      self._max_context)
        kv_bytes = 2 * n_layers * n_kv * head_dim * 2 * max_ctx * 64
        try:
            import torch
            vram = (torch.cuda.memory_allocated() / (1024 ** 3)
                    if torch.cuda.is_available() else 0.0)
        except ImportError:
            vram = 0.0
        return LoadReport(model_id=model_id, load_seconds=round(dt, 2),
                          vram_gb=round(vram, 2),
                          kv_cache_gb=round(kv_bytes / (1024 ** 3), 3),
                          max_context=max_ctx, backend=self.name)

    def unload(self) -> None:
        m, self._model = self._model, None
        self._tok = None
        self._model_id = None
        del m
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
        ids = self._tok(prompt, return_tensors="pt").to(self._model.device)
        out = self._model.generate(
            **ids, max_new_tokens=max_tokens,
            do_sample=temperature > 0, temperature=max(temperature, 1e-5),
            use_cache=True)  # full KV cache during generation
        return self._tok.decode(out[0][ids["input_ids"].shape[1]:],
                                skip_special_tokens=True)

    def chat_generate(self, messages: list[dict], max_tokens: int = 128,
                      temperature: float = 0.7, **kw) -> str:
        if hasattr(self._tok, "apply_chat_template"):
            prompt = self._tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = "\n".join(
                f"{m.get('role', 'user')}: {m.get('content', '')}"
                for m in messages) + "\nassistant:"
        return self.generate(prompt, max_tokens=max_tokens,
                             temperature=temperature, **kw)

    @property
    def loaded(self) -> bool:
        return self._model is not None

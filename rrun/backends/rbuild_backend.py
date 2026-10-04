"""R-Build backend — serve native R-Build checkpoints with zero conversion.

model_id is the path to a checkpoint directory written by
``Trainer.save_checkpoint`` (rbuild_config.json + model.pt + meta.json).
Loading goes through ``Trainer.load_checkpoint``, so the full v3 feature
set rides along: thinking modes, generation watermarking, fast-weight
memory, and the auto-train toggle (off by default while serving).

Tokenization: byte-level UTF-8 by default — matches rbuild's local .txt
training path (any vocab_size >= 256 works). Pass ``tokenizer="<hf-id>"``
to use a HuggingFace tokenizer instead (matches the HF-dataset training
path in examples/train_interactive.py).
"""
from __future__ import annotations

import gc
import time

from .base import Backend, LoadReport


class RBuildBackend(Backend):
    name = "rbuild"

    def __init__(self, device: str | None = None,
                 tokenizer: str | None = None,
                 watermark: bool | None = None,
                 max_context: int | None = None) -> None:
        self._device = device                  # None -> CPU (or Trainer default)
        self._tokenizer_name = tokenizer       # None -> byte-level UTF-8
        self._watermark = watermark            # None -> checkpoint's config
        self._max_context = max_context        # cap on the model's max_seq_len
        self._model = None
        self._cfg = None
        self._tok = None
        self._model_id: str | None = None

    # ------------------------------------------------------------ tokenize
    def _encode(self, text: str) -> list[int]:
        if self._tok is not None:
            return self._tok.encode(text)
        vocab = self._cfg.model.vocab_size
        return [b % vocab for b in text.encode("utf-8")]

    def _decode(self, ids: list[int]) -> str:
        if self._tok is not None:
            return self._tok.decode(ids, skip_special_tokens=True)
        return bytes(i % 256 for i in ids).decode("utf-8", errors="replace")

    # ---------------------------------------------------------------- load
    def load(self, model_id: str, **kw) -> LoadReport:
        from rbuild import Trainer            # same repo — no conversion

        t0 = time.perf_counter()
        self._model = Trainer.load_checkpoint(model_id, device=self._device)
        self._cfg = self._model.cfg
        if self._tokenizer_name is not None:
            from transformers import AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self._tokenizer_name)
        self._model_id = model_id
        dt = time.perf_counter() - t0

        cfg = self._cfg
        # R-Build geometry: extraction-loop blocks + every branch of every
        # parallel stage carries attention, so all of them hold KV state.
        n_layers = (cfg.cache_loop.n_layers
                    + cfg.parallel.n_stages * cfg.parallel.n_branches)
        n_kv_heads = cfg.model.n_kv_heads
        head_dim = cfg.model.resolved_head_dim()
        max_ctx = cfg.model.max_seq_len
        if self._max_context:
            max_ctx = min(max_ctx, self._max_context)
        kv_bytes = (2 * n_layers * n_kv_heads * head_dim * 2
                    * max_ctx * 64)
        weight_bytes = sum(p.numel() * p.element_size()
                           for p in self._model.parameters())
        weight_bytes += sum(b.numel() * b.element_size()
                            for b in self._model.buffers())
        return LoadReport(
            model_id=model_id, load_seconds=round(dt, 2),
            vram_gb=round(weight_bytes / (1024 ** 3), 3),
            kv_cache_gb=round(kv_bytes / (1024 ** 3), 3),
            max_context=max_ctx, backend=self.name,
            extra={"n_layers": n_layers, "n_kv_heads": n_kv_heads,
                   "head_dim": head_dim,
                   "total_params": cfg.count_parameters()["total_params"]})

    def unload(self) -> None:
        m, self._model = self._model, None
        self._cfg = None
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

    # ------------------------------------------------------------- generate
    def generate(self, prompt: str, max_tokens: int = 128,
                 temperature: float = 0.7, **kw) -> str:
        import torch

        ids = torch.tensor([self._encode(prompt)], dtype=torch.long,
                           device=self._device)
        watermark = kw.get("watermark", self._watermark)
        out = self._model.generate(ids, max_new_tokens=max_tokens,
                                   temperature=temperature,
                                   watermark=watermark)
        return self._decode(out[0, ids.shape[1]:].tolist())

    @property
    def loaded(self) -> bool:
        return self._model is not None

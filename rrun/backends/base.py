"""Backend interface. A backend owns exactly ONE loaded model at a time."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Optional


@dataclass
class LoadReport:
    model_id: str
    load_seconds: float
    vram_gb: float                       # model weights footprint
    kv_cache_gb: float                   # allocated KV cache footprint
    max_context: int
    backend: str
    extra: dict = field(default_factory=dict)


class Backend:
    """Uniform interface over rbuild / vLLM / HF transformers / mock.

    Contract:
      - load() pulls weights in and allocates a FULL KV cache for the model.
      - unload() must release weights AND KV cache completely (wipe), so the
        next model starts from a clean slate.
      - generate() / stream() serve inference.
    """

    name = "base"

    def load(self, model_id: str, **kw) -> LoadReport:
        raise NotImplementedError

    def unload(self) -> None:
        raise NotImplementedError

    def generate(self, prompt: str, max_tokens: int = 128,
                 temperature: float = 0.7, **kw) -> str:
        raise NotImplementedError

    def stream(self, prompt: str, max_tokens: int = 128,
               temperature: float = 0.7, **kw) -> Iterator[str]:
        yield self.generate(prompt, max_tokens=max_tokens,
                            temperature=temperature, **kw)

    def chat_generate(self, messages: list[dict], max_tokens: int = 128,
                      temperature: float = 0.7, **kw) -> str:
        prompt = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}"
                           for m in messages) + "\nassistant:"
        return self.generate(prompt, max_tokens=max_tokens,
                             temperature=temperature, **kw)

    @property
    def loaded(self) -> bool:
        raise NotImplementedError

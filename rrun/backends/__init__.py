from .base import Backend, LoadReport
from .mock_backend import MockBackend
from .rbuild_backend import RBuildBackend


def auto_backend(preference: str = "auto", **kw) -> Backend:
    """Pick a backend.

    Explicit: "rbuild" (native R-Build checkpoints), "vllm", "hf", "mock".
    "auto": vLLM when available (GPU hosts), else HF transformers, else mock.
    """
    if preference == "rbuild":
        return RBuildBackend(**{k: v for k, v in kw.items()
                                if k in ("device", "tokenizer",
                                         "watermark", "max_context")})
    if preference == "mock":
        return MockBackend(**{k: v for k, v in kw.items()
                              if k in ("simulated_vram_gb",
                                       "simulated_load_s", "max_context")})
    if preference in ("auto", "vllm"):
        try:
            import vllm  # noqa: F401
            from .vllm_backend import VLLMBackend
            return VLLMBackend(**kw)
        except ImportError:
            if preference == "vllm":
                raise
    if preference in ("auto", "hf"):
        try:
            import transformers  # noqa: F401
            from .hf_backend import HFBackend
            return HFBackend()
        except ImportError:
            if preference == "hf":
                raise
    return MockBackend()


__all__ = ["Backend", "LoadReport", "MockBackend", "RBuildBackend",
           "auto_backend"]

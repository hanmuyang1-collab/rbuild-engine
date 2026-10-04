"""OpenAI-compatible HTTP server + admin swap endpoints.

The server process NEVER restarts on swap — the port stays bound and
clients keep their connections; only the resident model changes.
"""
from __future__ import annotations

import time
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .engine import (ModelNotLoadedError, RRunEngine, SwapInProgressError)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = 128
    temperature: float = 0.7


class SwapRequest(BaseModel):
    model_id: str
    load_kw: dict = Field(default_factory=dict)


def create_app(engine: RRunEngine) -> FastAPI:
    app = FastAPI(title="R-Run", version="0.2.0")

    @app.get("/health")
    def health():
        return {"ok": True, "resident": engine.backend.loaded}

    @app.get("/v1/models")
    def models():
        cur = engine.status()["model"]
        data = ([{"id": cur["model_id"], "object": "model",
                  "created": int(time.time()), "owned_by": "rrun"}]
                if cur else [])
        return {"object": "list", "data": data}

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest):
        try:
            t0 = time.perf_counter()
            text = engine.chat([m.model_dump() for m in req.messages],
                               max_tokens=req.max_tokens,
                               temperature=req.temperature)
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": engine.status()["model"]["model_id"],
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant",
                                         "content": text}}],
                "usage": {"latency_s": round(time.perf_counter() - t0, 3)},
            }
        except ModelNotLoadedError as e:
            raise HTTPException(503, str(e))

    # ------------------------------------------------------------ admin API
    @app.get("/admin/status")
    def status():
        return engine.status()

    @app.post("/admin/swap")
    def swap(req: SwapRequest):
        try:
            report = engine.swap(req.model_id, **req.load_kw)
            return {"ok": True, "swapped_to": report.model_id,
                    "load_seconds": report.load_seconds,
                    "kv_cache_gb": report.kv_cache_gb,
                    "kv_cache_full": True,
                    "max_context": report.max_context}
        except SwapInProgressError as e:
            raise HTTPException(409, str(e))
        except Exception as e:  # load failure — old model already wiped
            raise HTTPException(500, f"swap failed after wipe: {e}")

    @app.post("/admin/wipe")
    def wipe():
        engine.kv.wipe()
        engine.backend.unload()
        engine._current = None
        return {"ok": True, "resident": False}

    return app


def run_server(engine: RRunEngine, host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(create_app(engine), host=host, port=port,
                log_level="info")

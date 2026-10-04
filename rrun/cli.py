"""rrun — one-command model hosting & swapping.

    rrun serve <model-id-or-path>     # host a model (starts the server)
    rrun swap  <model-id-or-path>     # wipe resident model, host a new one
    rrun status                       # what's resident + KV cache state
    rrun wipe                         # unload everything, free VRAM

`rrun swap X` is the single command that replaces the running model with
zero server restart and a full KV cache for the new model.

Backends (`--backend`):
    rbuild   native R-Build checkpoints (Trainer.save_checkpoint dirs)
    vllm     vLLM on GPU hosts
    hf       HuggingFace transformers (CPU-friendly)
    mock     no weights — smoke tests
    auto     vllm > hf > mock
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import urllib.error

DEFAULT_PORT = 8000


def _post(port: int, path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=None) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        print(f"server error {e.code}: {detail}", file=sys.stderr)
        sys.exit(1)


def _get(port: int, path: str) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}",
                                timeout=10) as r:
        return json.loads(r.read())


def _server_up(port: int) -> bool:
    try:
        _get(port, "/health")
        return True
    except Exception:
        return False


def _launch_server(model_id: str, args) -> None:
    from .backends import auto_backend
    from .engine import RRunEngine
    from .server import run_server

    backend = auto_backend(args.backend)
    engine = RRunEngine(backend, max_context=args.max_context)
    engine.serve(model_id)
    print(f"[rrun] {model_id} resident — full KV cache "
          f"({engine.kv.status().get('gb', '?')} GB), "
          f"context {engine.status()['model']['max_context']}")
    run_server(engine, host=args.host, port=args.port)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="rrun", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--backend", default="auto",
                   choices=["auto", "rbuild", "vllm", "hf", "mock"])
    p.add_argument("--max-context", type=int, default=32768)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="host a model (cold start)")
    s.add_argument("model_id")
    w = sub.add_parser("swap", help="wipe resident model and host a new one")
    w.add_argument("model_id")
    sub.add_parser("status")
    sub.add_parser("wipe")

    args = p.parse_args(argv)

    if args.cmd == "serve":
        if _server_up(args.port):
            print("[rrun] server already running — "
                  "use `rrun swap` instead", file=sys.stderr)
            sys.exit(2)
        _launch_server(args.model_id, args)

    elif args.cmd == "swap":
        if not _server_up(args.port):
            # nothing running: swap degrades gracefully into serve
            print("[rrun] no server running — starting one")
            _launch_server(args.model_id, args)
            return
        print(f"[rrun] swapping to {args.model_id} ...")
        out = _post(args.port, "/admin/swap", {"model_id": args.model_id})
        print(f"[rrun] now serving {out['swapped_to']} — "
              f"loaded in {out['load_seconds']}s, "
              f"full KV cache {out['kv_cache_gb']} GB, "
              f"context {out['max_context']}")

    elif args.cmd == "status":
        print(json.dumps(_get(args.port, "/admin/status"), indent=2))

    elif args.cmd == "wipe":
        out = _post(args.port, "/admin/wipe", {})
        print(f"[rrun] wiped — resident: {out['resident']}")


if __name__ == "__main__":
    main()

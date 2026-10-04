"""R-Run smoke test — proves the swap contract end to end on any machine
(mock backend, no GPU / no weights needed; rbuild backend on tiny native
checkpoints):

  1. serve model A            -> resident + full KV cache
  2. generate from A          -> works
  3. swap A -> B              -> A fully wiped, B resident, FULL new KV cache
  4. generate from B          -> works, server/engine never restarted
  5. swap B -> C              -> repeatable
  6. HTTP layer               -> /admin/swap + /v1/chat/completions consistent
  7. rbuild backend           -> native checkpoints serve + swap, KV sized
                                 from R-Build geometry, counter-verified

Run:  python tests/smoke_test.py        (or: pytest tests/smoke_test.py)
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rrun.backends import MockBackend                      # noqa: E402
from rrun.engine import RRunEngine                          # noqa: E402

MODEL_A = "GeoThinkAI/R-build-20b-a3.4b"
MODEL_B = "Qwen/Qwen3-32B"
MODEL_C = "GeoThinkAI/R-build-90b-a8.9b"

passed = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        raise AssertionError(name)
    passed.append(name)


def test_engine_swap():
    print("== engine swap contract ==")
    eng = RRunEngine(MockBackend(simulated_vram_gb=18.5),
                     max_context=131072, max_seqs=64)

    # 1. cold serve A
    rep_a = eng.serve(MODEL_A)
    check("serve A resident", eng.backend.loaded)
    check("A full KV cache", eng.kv.status()["full"] is True,
          f"{eng.kv.status()['gb']} GB @ {rep_a.max_context} ctx")

    # 2. inference on A
    out_a = eng.generate("hello from A")
    check("generate from A", MODEL_A in out_a, out_a[:60])

    # 3. swap A -> B
    wipes_before = eng.kv.wipe_count
    t0 = time.perf_counter()
    rep_b = eng.swap(MODEL_B)
    swap_s = time.perf_counter() - t0
    st = eng.status()
    check("swap wiped old cache", eng.kv.wipe_count == wipes_before + 1)
    check("B is resident", st["model"]["model_id"] == MODEL_B)
    check("A gone from cache", st["kv_cache"]["model_id"] == MODEL_B)
    check("B full KV cache", st["kv_cache"]["full"] is True,
          f"{st['kv_cache']['gb']} GB")
    check("swap was fast", swap_s < 2.0, f"{swap_s*1000:.0f} ms (mock)")

    # 4. inference on B, same engine object (no restart)
    out_b = eng.generate("hello from B")
    check("generate from B", MODEL_B in out_b)

    # 5. second swap B -> C, history accumulates
    eng.swap(MODEL_C)
    check("repeatable swap", eng.status()["model"]["model_id"] == MODEL_C)
    check("swap history", eng.status()["swaps"] == 2)


def test_http_layer():
    print("== http layer ==")
    from fastapi.testclient import TestClient
    from rrun.server import create_app

    eng = RRunEngine(MockBackend())
    eng.serve(MODEL_A)
    c = TestClient(create_app(eng))

    r = c.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "ping"}]})
    check("chat 200", r.status_code == 200)
    check("chat answered by A", MODEL_A in
          r.json()["choices"][0]["message"]["content"])

    r = c.post("/admin/swap", json={"model_id": MODEL_B})
    check("admin swap 200", r.status_code == 200)
    body = r.json()
    check("admin swap reports full KV", body["kv_cache_full"] is True,
          f"{body['kv_cache_gb']} GB")

    r = c.get("/v1/models")
    check("/v1/models shows B only",
          [m["id"] for m in r.json()["data"]] == [MODEL_B])

    r = c.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "ping again"}]})
    check("post-swap chat served by B", MODEL_B in
          r.json()["choices"][0]["message"]["content"])


def test_rbuild_backend():
    """Native R-Build checkpoints through the same serve/swap contract."""
    print("== rbuild backend (native checkpoints) ==")
    import tempfile
    import torch
    from rbuild import RBuildConfig, RBuildModel, Trainer
    from rrun.backends.rbuild_backend import RBuildBackend

    def make_ckpt(path: str, seed: int) -> RBuildConfig:
        cfg = RBuildConfig()
        cfg.model.max_seq_len = 64
        cfg.critic.max_loops = 2
        torch.manual_seed(seed)
        model = RBuildModel(cfg)
        actual = sum(p.numel() for p in model.parameters())
        assert actual == cfg.count_parameters()["total_params"]
        Trainer(model, cfg, device="cpu", log_fn=None).save_checkpoint(path)
        return cfg

    root = tempfile.mkdtemp(prefix="rrun_rbuild_")
    ckpt_a, ckpt_b = os.path.join(root, "ckpt_a"), os.path.join(root, "ckpt_b")
    cfg_a = make_ckpt(ckpt_a, 1)
    make_ckpt(ckpt_b, 2)

    eng = RRunEngine(RBuildBackend(max_context=64), max_context=64, max_seqs=4)

    # serve checkpoint A — KV cache sized from R-Build geometry
    rep_a = eng.serve(ckpt_a)
    expect_layers = (cfg_a.cache_loop.n_layers
                     + cfg_a.parallel.n_stages * cfg_a.parallel.n_branches)
    check("rbuild A resident", eng.backend.loaded)
    check("KV from R-Build geometry",
          rep_a.extra["n_layers"] == expect_layers
          and rep_a.extra["n_kv_heads"] == cfg_a.model.n_kv_heads
          and rep_a.extra["head_dim"] == cfg_a.model.resolved_head_dim(),
          f"{expect_layers} effective layers")
    check("counter-verified load",
          rep_a.extra["total_params"]
          == cfg_a.count_parameters()["total_params"])
    check("rbuild full KV cache", eng.kv.status()["full"] is True)

    # generate (byte-level path), watermark kwarg accepted
    out_a = eng.generate("hello from A", max_tokens=8, watermark=False)
    check("generate from rbuild A", isinstance(out_a, str) and len(out_a) > 0)

    # swap checkpoint A -> B, same engine, full new KV cache
    wipes_before = eng.kv.wipe_count
    eng.swap(ckpt_b)
    st = eng.status()
    check("rbuild swap wiped old cache",
          eng.kv.wipe_count == wipes_before + 1)
    check("rbuild B resident", st["model"]["model_id"] == ckpt_b)
    check("rbuild B full KV cache", st["kv_cache"]["full"] is True)
    out_b = eng.generate("hello from B", max_tokens=8)
    check("generate from rbuild B", isinstance(out_b, str) and len(out_b) > 0)

    # HTTP layer over the native backend
    from fastapi.testclient import TestClient
    from rrun.server import create_app
    c = TestClient(create_app(eng))
    r = c.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 4})
    check("rbuild chat 200", r.status_code == 200)
    r = c.post("/admin/swap", json={"model_id": ckpt_a})
    check("rbuild admin swap 200", r.status_code == 200)
    check("rbuild admin swap reports full KV",
          r.json()["kv_cache_full"] is True)


if __name__ == "__main__":
    test_engine_swap()
    test_http_layer()
    test_rbuild_backend()
    print(f"\nSMOKE TEST PASSED — {len(passed)} checks green")

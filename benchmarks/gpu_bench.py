"""R-Build GPU benchmark: training throughput, peak memory and decode speed.

Runs unchanged against v4.2 and v4.3 so the two can be compared on the same
GPU. Usage:  PYTHONPATH=. python benchmarks/gpu_bench.py [--seq 2048] [--batch 4]
"""
import argparse, inspect, json, time
import torch
from rbuild import RBuildConfig, RBuildModel


def build_cfg(a):
    cfg = RBuildConfig()
    m = cfg.model
    m.vocab_size, m.d_model, m.n_heads, m.n_kv_heads = a.vocab, a.d_model, a.d_model // 64, 4
    m.max_seq_len = a.seq + 256
    cfg.cache_loop.n_layers, cfg.cache_loop.n_loops = 2, 3
    cfg.parallel.n_stages, cfg.parallel.n_branches = 3, 2
    cfg.parallel.n_experts, cfg.parallel.expert_top_k = 16, 2
    cfg.memory.key_dim, cfg.memory.value_dim = 128, a.d_model
    for sec, name in [("draft_verify", "enabled"), ("noting", "enabled"),
                      ("outgen", "enabled"), ("vision", "enabled")]:
        if hasattr(cfg, sec) and hasattr(getattr(cfg, sec), name):
            setattr(getattr(cfg, sec), name, False)
    if hasattr(cfg.train, "grad_checkpoint"):
        cfg.train.grad_checkpoint = a.ckpt
    cfg.validate()
    return cfg


def train_bench(a, cfg):
    torch.manual_seed(0)
    model = RBuildModel(cfg).cuda().train()
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randint(0, a.vocab, (a.batch, a.seq), device="cuda")
    y = torch.randint(0, a.vocab, (a.batch, a.seq), device="cuda")

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        return loss

    for _ in range(2):
        step()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t = time.time()
    for _ in range(a.steps):
        loss = step()
    torch.cuda.synchronize()
    dt = (time.time() - t) / a.steps
    return {"params_M": round(n_params / 1e6, 1),
            "step_ms": round(dt * 1000, 1),
            "tokens_per_s": round(a.batch * a.seq / dt),
            "peak_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "loss": round(float(loss), 4)}


@torch.no_grad()
def decode_bench(a, cfg):
    torch.manual_seed(0)
    model = RBuildModel(cfg).cuda().eval()
    p = torch.randint(0, a.vocab, (1, a.prompt), device="cuda")
    out = {}
    modes = [False]
    if "kv_cache" in inspect.signature(model.generate).parameters:
        modes.append(True)
    for kv in modes:
        kw = {"kv_cache": True} if kv else {}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model.generate(p, max_new_tokens=8, top_k=1, **kw)
            torch.cuda.synchronize(); t = time.time()
            model.generate(p, max_new_tokens=a.new, top_k=1, **kw)
            torch.cuda.synchronize()
        out["kv_cache" if kv else "re_encode"] = round(a.new / (time.time() - t), 1)
    return {"decode_tokens_per_s": out}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--d_model", type=int, default=1024)
    ap.add_argument("--vocab", type=int, default=32000)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--ckpt", action="store_true")
    ap.add_argument("--prompt", type=int, default=512)
    ap.add_argument("--new", type=int, default=128)
    a = ap.parse_args()
    import rbuild
    cfg = build_cfg(a)
    res = {"version": rbuild.__version__, "gpu": torch.cuda.get_device_name(),
           "args": vars(a)}
    try:
        res["train"] = train_bench(a, cfg)
    except torch.cuda.OutOfMemoryError:
        res["train"] = "OOM"
    torch.cuda.empty_cache()
    res.update(decode_bench(a, cfg))
    print(json.dumps(res))

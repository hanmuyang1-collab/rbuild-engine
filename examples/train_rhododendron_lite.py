#!/usr/bin/env python3
"""
train_rhododendron_lite.py — train rhododendron-lite (20B-A3.5B, *blind*)
on the R-Build v2.1 engine.

Blind = text-only: vision.enabled=False, so no vision tower is built and the
run is bit-identical in shape to the text-only path. (Flip vision on later
for the s2+ VL stage without changing this script's structure.)

Config (counter-verified, tunable via CLI overrides):
  d_model=2048, 16 heads / 4 kv, cache-loop 2 layers x 8 loops (pulls every loop),
  7 parallel stages x 6 branches (gate-bundled), 128 experts top-8 + shared,
  expert_dim=512, MoD 0.5 (cache loop) / 0.75 (parallel), vocab=128000
  -> 19.87B total / 3.47B active per token

Data streams from HuggingFace (no full download) — default FineWeb-Edu sample.
Everything prints the naive-vs-optimized cost report before starting.

Single GPU:
  python train_rhododendron_lite.py --smoke
8x H100 (FSDP):
  torchrun --nproc_per_node=8 train_rhododendron_lite.py --distributed fsdp \
      --max-steps 100000 --batch-size 8 --grad-accum 8
"""

from __future__ import annotations

import argparse
import os
import time

import torch


# --------------------------------------------------------------------------- #
# the model config — rhododendron-lite, 20B-A3.5B, blind
# --------------------------------------------------------------------------- #

def build_config(vocab_size: int, args) -> "object":
    from rbuild import preset

    cfg = preset("s1")                      # 19.9B/3.3B starting point
    # tuned to hit 20B-A3.5B exactly (counter-verified)
    cfg.cache_loop.n_loops = 8
    cfg.parallel.mod_capacity = 0.75
    # ---- blind: no vision tower at all
    cfg.vision.enabled = False
    # ---- tokenizer-derived
    cfg.model.vocab_size = vocab_size
    cfg.model.max_seq_len = args.seq_len
    # ---- training stack
    cfg.train.max_steps = args.max_steps
    cfg.train.batch_size = args.batch_size
    cfg.train.grad_accum = args.grad_accum
    cfg.train.precision = args.precision
    cfg.train.warmup_steps = args.warmup_steps
    cfg.train.cooldown_frac = args.cooldown_frac
    cfg.train.grad_checkpoint = args.grad_checkpoint
    cfg.train.lr = args.lr
    cfg.train.adamw_lr = args.adamw_lr
    cfg.train.gpu_price_per_hour = args.gpu_price
    cfg.train.gpus = args.gpus
    return cfg


# --------------------------------------------------------------------------- #
# streaming data (no full dataset download)
# --------------------------------------------------------------------------- #

def stream_batches(tokenizer, args, rank: int = 0, world: int = 1):
    """
    Streams text from the Hub, tokenizes, packs into (seq_len+1) windows,
    yields (input_ids, targets) with targets shifted by one.
    Re-iterates forever; each rank skips ahead so shards don't overlap.
    """
    from datasets import load_dataset

    ds = load_dataset(args.dataset, name=args.dataset_name,
                      split="train", streaming=True)
    if world > 1:
        ds = ds.shard(num_shards=world, index=rank)

    T = args.seq_len + 1
    buf = []
    for row in ds:
        text = row.get("text") or row.get("content") or ""
        if not text:
            continue
        buf.extend(tokenizer.encode(text, add_special_tokens=False))
        while len(buf) >= T:
            window = torch.tensor(buf[:T], dtype=torch.long)
            buf = buf[T:]
            yield window[:-1], window[1:]


def batch_iterator(tokenizer, args, rank: int = 0, world: int = 1):
    """Groups the token stream into batches of cfg.train.batch_size."""
    gen = stream_batches(tokenizer, args, rank, world)
    while True:
        xs, ys = [], []
        for _ in range(args.batch_size):
            x, y = next(gen)
            xs.append(x); ys.append(y)
        yield torch.stack(xs), torch.stack(ys)


# --------------------------------------------------------------------------- #
# distributed setup
# --------------------------------------------------------------------------- #

def setup_distributed(args):
    if args.distributed == "none":
        return 0, 1
    import torch.distributed as dist
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    return rank, world


def make_wrap_fn(args):
    if args.distributed == "fsdp":
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision
        mp = MixedPrecision(param_dtype=torch.bfloat16,
                            reduce_dtype=torch.bfloat16,
                            buffer_dtype=torch.bfloat16) if args.precision != "fp32" else None

        def wrap(model):
            return FSDP(model, use_orig_params=True, mixed_precision=mp)
        return wrap
    if args.distributed == "ddp":
        from torch.nn.parallel import DistributedDataParallel as DDP

        def wrap(model):
            return DDP(model, device_ids=[torch.cuda.current_device()])
        return wrap
    return None


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Train rhododendron-lite (20B-A3.5B, blind)")
    ap.add_argument("--tokenizer", default="NousResearch/Llama-2-7b-hf",
                    help="HF tokenizer (vocab is taken from it; 128000-token vocab recommended)")
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--dataset-name", default="sample-10BT")
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--max-steps", type=int, default=100000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--precision", choices=["fp32", "bf16", "fp8"], default="bf16")
    ap.add_argument("--warmup-steps", type=int, default=2000)
    ap.add_argument("--cooldown-frac", type=float, default=0.4)
    ap.add_argument("--lr", type=float, default=3e-3, help="Muon lr")
    ap.add_argument("--adamw-lr", type=float, default=3e-4)
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--distributed", choices=["none", "ddp", "fsdp"], default="none")
    ap.add_argument("--gpu-price", type=float, default=2.0, help="$/GPU-hour for the cost report")
    ap.add_argument("--gpus", type=int, default=8, help="GPU count for the cost report")
    ap.add_argument("--out", default="ckpt_rhododendron_lite")
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--resume", default=None, help="checkpoint dir to resume from")
    ap.add_argument("--push-to-hub", default=None,
                    help="e.g. GeoThinkAI/rhododendron-lite-20b-a3.5b — pushes final weights")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny model + 20 steps on random tokens (no download, sanity run)")
    args = ap.parse_args()

    from rbuild import RBuildConfig, RBuildModel, Trainer

    rank, world = setup_distributed(args)
    is_main = rank == 0

    # ---------------- config ---------------- #
    if args.smoke:
        cfg = RBuildConfig()                # tiny built-in config
        cfg.vision.enabled = False          # still blind
        cfg.train.max_steps = 20
        cfg.train.batch_size = 2
        cfg.train.grad_accum = 1
        cfg.train.warmup_steps = 2
        tokenizer = None
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        cfg = build_config(len(tokenizer), args)

    if is_main:
        print(cfg.report())
        print(f"\nrhododendron-lite: {cfg.count_parameters()['total_params']/1e9:.2f}B total / "
              f"{cfg.count_parameters()['active_params_per_token']/1e9:.2f}B active — blind\n")

    # ---------------- model ---------------- #
    if args.resume:
        model = Trainer.load_checkpoint(args.resume,
                                        device="cuda" if torch.cuda.is_available() else "cpu")
        model.cfg = cfg                     # keep training-side overrides
    else:
        model = RBuildModel(cfg)
    actual = sum(p.numel() for p in model.parameters())
    expected = cfg.count_parameters()["total_params"]
    assert actual == expected, f"counter mismatch: built {actual}, counted {expected}"
    if is_main:
        print(f"counter check passed: {actual:,} params (exact match)")

    trainer = Trainer(model, cfg, log_fn=print if is_main else None,
                      wrap_fn=make_wrap_fn(args))

    # ---------------- data ---------------- #
    if args.smoke:
        def _batches():
            while True:
                yield (torch.randint(0, cfg.model.vocab_size, (2, 32)),
                       torch.randint(0, cfg.model.vocab_size, (2, 32)))
        batches = _batches()
    else:
        batches = batch_iterator(tokenizer, args, rank, world)

    # ---------------- train ---------------- #
    t0 = time.time()
    history = trainer.fit(batches, max_steps=cfg.train.max_steps,
                          save_every=args.save_every, save_dir=args.out)

    if is_main:
        trainer.save_checkpoint(args.out)
        print(f"\ndone in {(time.time()-t0)/3600:.2f} h -> {args.out}")

    # ---------------- optional hub push ---------------- #
    if args.push_to_hub and is_main:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.push_to_hub, exist_ok=True)
        api.upload_folder(folder_path=args.out, repo_id=args.push_to_hub)
        print(f"pushed -> https://huggingface.co/{args.push_to_hub}")

    if args.distributed != "none":
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

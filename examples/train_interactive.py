#!/usr/bin/env python3
"""
R-Build v3 — the completely interactive trainer (tutorial edition).

Anyone can run this — no R-Build knowledge needed:

    python examples/train_interactive.py

The script walks you through TEN numbered steps. Every question shows the
recommended value in [brackets] — just press Enter to accept it. Every
answer is validated, and invalid input is re-asked with an explanation.

    1. model size (preset ladder)        6. native actuation
    2. critic experts + ACT halting      7. training knobs
    3. noting + self-training            8. data source
    4. thinking mode                     9. review + train
    5. vision (blind / ViT / encoderless) 10. save (+ optional HF push)

Don't want prompts? Edit the ANSWERS dict right below this docstring — any
key you fill in is used instead of asking (set AUTO = True to skip ALL
prompts and run on defaults/your edits).
"""

from __future__ import annotations

import os
import sys
import time

import torch

# --------------------------------------------------------------------- #
# EDIT ME — non-interactive answers (leave a key as None to be asked)   #
# --------------------------------------------------------------------- #

AUTO = False                      # True: no prompts at all, use these values
ANSWERS = {
    "preset": None,               # "tiny" | "s1" | "s2" | "s3" | "s4" | "s5"
    "critics": None,              # True/False
    "n_critics": None,            # X critics per panel (int)
    "y_critics": None,            # Y critics needed to halt (int)
    "max_loops": None,            # extraction-loop cap (int)
    "noting": None,               # True/False (critic-verified self-training)
    "thinking_mode": None,        # "fast"|"balanced"|"deep"|"careful"|"research"|"custom"
    "vision": None,               # "blind" | "vit" | "encoderless"
    "vawu": None,                 # True/False (whole-video tokens)
    "actuation": None,            # True/False (native action tokens)
    "steps": None,                # training steps (int)
    "batch_size": None,           # sequences per step (int)
    "seq_len": None,              # tokens per sequence (int)
    "precision": None,            # "fp32" | "bf16" | "fp8"
    "self_train_every": None,     # consolidate notes every N steps (0 = off)
    "data": None,                 # "toy" | "text" | "hf"
    "data_path": None,            # path for "text" (local .txt file)
    "save_dir": None,             # checkpoint directory
    "hf_repo": None,              # e.g. "GeoThinkAI/R-build-20b-a3.4b" ("" = skip)
    "hf_token": None,             # "hf_..." token ("" = skip)  <-- PUT YOURS HERE
}

# --------------------------------------------------------------------- #
# input machinery — numbered, validated, tutorial-friendly
# --------------------------------------------------------------------- #

_step = 0


def ask(key, question, default, cast=str, choices=None, hint=""):
    """
    Ask one question. `default` is taken on Enter. `choices` (if given)
    restricts answers; `cast` converts the input. Re-asks on bad input.
    """
    global _step
    if AUTO or ANSWERS.get(key) is not None:
        return cast(ANSWERS[key]) if ANSWERS.get(key) is not None else default
    _step += 1
    opts = f"  options: {' / '.join(map(str, choices))}" if choices else ""
    if hint:
        print(f"      {hint}")
    while True:
        raw = input(f"  [{_step:02d}] {question} [{default}]{opts}: ").strip()
        if raw == "":
            return default
        try:
            val = cast(raw)
        except (ValueError, TypeError):
            print(f"        ! needs to be {cast.__name__} — try again")
            continue
        if choices and val not in choices:
            print(f"        ! pick one of {choices}")
            continue
        return val


def ask_bool(key, question, default, hint=""):
    return ask(key, question + " (y/n)", "y" if default else "n",
               cast=lambda s: s.lower() in ("y", "yes", "1", "true"),
               hint=hint)


# --------------------------------------------------------------------- #
# data sources
# --------------------------------------------------------------------- #

def toy_batches(vocab_size, batch, seq):
    """Random tokens — zero downloads, instant sanity training."""
    while True:
        x = torch.randint(0, vocab_size, (batch, seq))
        yield x, x.clone()


def text_batches(path, vocab_size, batch, seq):
    """Local .txt file, byte-encoded (any vocab_size >= 256 works)."""
    with open(path, "rb") as f:
        data = f.read()
    ids = torch.tensor(list(data), dtype=torch.long)
    print(f"      loaded {len(ids):,} bytes from {path}")
    while True:
        starts = torch.randint(0, max(1, len(ids) - seq - 1), (batch,))
        x = torch.stack([ids[s:s + seq] for s in starts])
        y = torch.stack([ids[s + 1:s + seq + 1] for s in starts])
        yield x, y


def hf_batches(tokenizer_name, dataset_name, vocab_size, batch, seq):
    """HuggingFace streaming — no full download. Needs: pip install .[train]"""
    from transformers import AutoTokenizer
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    ds = load_dataset(dataset_name, split="train", streaming=True)
    buf = []
    for row in ds:
        buf.extend(tok(row["text"], add_special_tokens=False)["input_ids"])
        while len(buf) >= (seq + 1) * batch:
            chunk, buf = buf[:(seq + 1) * batch], buf[(seq + 1) * batch:]
            t = torch.tensor(chunk, dtype=torch.long).view(batch, seq + 1)
            yield t[:, :-1].clamp_max(vocab_size - 1), t[:, 1:].clamp_max(vocab_size - 1)


# --------------------------------------------------------------------- #
# the tutorial
# --------------------------------------------------------------------- #

def main():
    from rbuild import RBuildConfig, RBuildModel, Trainer, preset

    print("=" * 60)
    print("  R-Build v3 — interactive trainer (tutorial edition)")
    print("  Press Enter to accept any [recommended] value.")
    print("=" * 60)

    # ---- 1. model size ------------------------------------------------ #
    print("\nSTEP 1 — model size (the continued-training ladder)")
    print("      tiny = 33M, runs on CPU, for learning the ropes")
    print("      s1 = 20.0B-A3.5B (rhododendron-lite) — first real stage")
    print("      s2..s5 = the ladder up to 415B (needs serious GPUs)")
    name = ask("preset", "preset", "tiny",
               choices=["tiny", "s1", "s2", "s3", "s4", "s5"])
    cfg = preset(name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if name != "tiny" and device == "cpu":
        print("      ! no GPU detected — a ladder preset on CPU will be very slow.")

    # ---- 2. critics + ACT halting ------------------------------------- #
    print("\nSTEP 2 — critic experts + adaptive halting (the v3 core)")
    print("      X critics judge the model's own work; the extraction loop")
    print("      runs until Y of them are satisfied (ACT-style halting).")
    cfg.critic.enabled = ask_bool("critics", "enable critic experts", True)
    if cfg.critic.enabled:
        cfg.critic.n_critics = ask("n_critics", "X — critics per panel", 4, int)
        cfg.critic.y_critics = ask("y_critics", "Y — critics needed to halt",
                                   2, int,
                                   hint=f"must be <= X ({cfg.critic.n_critics})")
        cfg.critic.max_loops = ask("max_loops", "extraction-loop cap",
                                   cfg.critic.max_loops, int)

    # ---- 3. noting + self-training ------------------------------------ #
    print("\nSTEP 3 — noting experts + critic-verified self-training")
    print("      the model takes notes while running; critics verify them;")
    print("      verified notes enter fast-weight memory instantly (no training)")
    print("      and queue for consolidation into the slow weights.")
    cfg.noting.enabled = cfg.critic.enabled and ask_bool(
        "noting", "enable self-training-while-running", True)

    # ---- 4. thinking mode --------------------------------------------- #
    print("\nSTEP 4 — thinking mode (retunes loops/critics/sampling live)")
    cfg.thinking.default_mode = ask(
        "thinking_mode", "mode", "balanced",
        choices=["fast", "balanced", "deep", "careful", "research"],
        hint="fast = 2 loops; research = 24 loops + strictest critics")

    # ---- 5. vision ----------------------------------------------------- #
    print("\nSTEP 5 — vision (training below stays text; this shapes the model)")
    vmode = ask("vision", "vision", "blind",
                choices=["blind", "vit", "encoderless"],
                hint="encoderless = v3, no vision encoder at all")
    if vmode != "blind":
        cfg.vision.enabled = True
        cfg.vision.mode = vmode
        cfg.vision.image_token_id = cfg.model.vocab_size - 1
        cfg.vision.vawu = ask_bool("vawu", "VaWU whole-video tokens", True)

    # ---- 6. actuation -------------------------------------------------- #
    print("\nSTEP 6 — native actuation (model clicks by generating tokens)")
    cfg.actuation.enabled = ask_bool("actuation", "enable action tokens", False,
                                     hint="adds click/scroll/type/wait tokens")
    if cfg.actuation.enabled:
        cfg.actuation.screen_grid = ask("screen_grid", "click grid size", 32, int)

    # ---- 7. training knobs --------------------------------------------- #
    print("\nSTEP 7 — training knobs")
    cfg.train.max_steps = ask("steps", "training steps", 50 if name == "tiny" else 2000, int)
    cfg.train.batch_size = ask("batch_size", "batch size", 4 if name == "tiny" else 8, int)
    cfg.model.max_seq_len = ask("seq_len", "sequence length", 128 if name == "tiny" else 2048, int)
    cfg.train.precision = ask("precision", "precision",
                              "fp32" if device == "cpu" else "bf16",
                              choices=["fp32", "bf16", "fp8"])
    cfg.train.self_train_every = ask(
        "self_train_every", "consolidate verified notes every N steps (0 = off)",
        0 if not cfg.noting.enabled else 25, int)
    cfg.train.grad_accum = 1

    # ---- 8. data -------------------------------------------------------- #
    print("\nSTEP 8 — data source")
    data_kind = ask("data", "data", "toy", choices=["toy", "text", "hf"],
                    hint="toy = random tokens, instant; text = local .txt; hf = streaming")
    if data_kind == "text":
        path = ask("data_path", "path to .txt file", "data.txt")
        while not os.path.exists(path):
            path = ask("data_path", f"not found — path to .txt file", "data.txt")
        batches = text_batches(path, cfg.effective_vocab_size(),
                               cfg.train.batch_size, cfg.model.max_seq_len)
    elif data_kind == "hf":
        tok_name = ask("hf_tokenizer", "tokenizer (HF id)", "gpt2")
        ds_name = ask("hf_dataset", "dataset (HF id)", "wikitext",
                      hint="streams — no full download")
        batches = hf_batches(tok_name, ds_name, cfg.effective_vocab_size(),
                             cfg.train.batch_size, cfg.model.max_seq_len)
    else:
        batches = toy_batches(cfg.effective_vocab_size(),
                              cfg.train.batch_size, cfg.model.max_seq_len)

    # ---- 9. review + build + train ------------------------------------- #
    print("\nSTEP 9 — review")
    print(cfg.report())
    if not ask_bool("confirm", "build the model and train", True):
        print("aborted — nothing was built.")
        return

    print("\nbuilding model ...")
    model = RBuildModel(cfg)
    n = sum(p.numel() for p in model.parameters())
    counted = cfg.count_parameters()["total_params"]
    assert n == counted, f"counter mismatch: {n} vs {counted}"
    print(f"      counter-verified: {n:,} params == counted {counted:,}")
    model.thinking_mode.apply(cfg.thinking.default_mode)
    print(f"      thinking mode: {cfg.thinking.default_mode}")

    trainer = Trainer(model, cfg, device=device)
    print(f"\ntraining on {device} for {cfg.train.max_steps} steps ...\n")
    t0 = time.time()
    hist = trainer.fit(batches, max_steps=cfg.train.max_steps)
    mins = (time.time() - t0) / 60
    print(f"\ndone in {mins:.1f} min — final loss {hist['loss'][-1]:.4f}")
    print(f"self-learning: {model.self_learn_stats()}")
    if model.cache_loop.last_halting:
        print(f"adaptive halting: {model.cache_loop.last_halting}")

    # ---- 10. save (+ optional HF push) ---------------------------------- #
    print("\nSTEP 10 — save")
    save_dir = ask("save_dir", "checkpoint directory", "ckpt_v3")
    trainer.save_checkpoint(save_dir)

    hf_repo = ask("hf_repo", "push to HuggingFace repo (blank = skip)", "")
    if hf_repo:
        token = ask("hf_token", "HF token (hf_...)", "")
        if token:
            from huggingface_hub import HfApi
            HfApi(token=token).upload_folder(folder_path=save_dir, repo_id=hf_repo)
            print(f"      pushed -> https://huggingface.co/{hf_repo}")
        else:
            print("      no token — skipped (checkpoint is safe on disk).")

    print("\nall done. reload with:  from rbuild import Trainer; "
          f"model = Trainer.load_checkpoint('{save_dir}')")


if __name__ == "__main__":
    main()

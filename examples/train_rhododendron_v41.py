#!/usr/bin/env python3
"""Rhododendron V4.1 continued-pretraining / supervised-fine-tuning example.

Qwen3-235B-A22B is the selectable default for THIS PROJECT ONLY. R-Build does
not require Qwen: replace --base-model with any compatible causal-LM checkpoint.

CPT:
  python examples/train_rhododendron_v41.py --mode cpt \
      --dataset HuggingFaceFW/fineweb-edu --dataset-name sample-10BT

SFT (a streaming dataset with prompt/response columns):
  python examples/train_rhododendron_v41.py --mode sft \
      --dataset your/rhododendron-sft --prompt-column prompt --response-column response

The 235B checkpoint requires a suitably sharded multi-GPU environment. This
script deliberately does not pretend a single small GPU can train it.
"""

from __future__ import annotations

import argparse

import torch


class HFRows:
    """Re-iterable streaming rows, so WSD runs can span multiple epochs."""

    def __init__(self, args):
        self.args = args

    def __iter__(self):
        from datasets import load_dataset

        args = self.args
        dataset = load_dataset(
            args.dataset,
            name=args.dataset_name,
            split=args.split,
            streaming=True,
        )
        for row in dataset:
            if args.mode == "cpt":
                text = row.get(args.text_column) or row.get("content") or ""
                if text:
                    yield text
            else:
                prompt = row.get(args.prompt_column)
                response = row.get(args.response_column)
                if prompt is not None and response is not None:
                    yield {"prompt": str(prompt), "response": str(response)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["cpt", "sft"], required=True)
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3-235B-A22B",
        help="Rhododendron default only; any compatible base remains selectable",
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--prompt-column", default="prompt")
    parser.add_argument("--response-column", default="response")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--cooldown-frac", type=float, default=0.4)
    parser.add_argument("--lr", type=float, default=3e-4, help="Muon learning rate")
    parser.add_argument("--adamw-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp8"], default="bf16")
    parser.add_argument("--freeze-base", action="store_true", help="train R-Build adapters only")
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--no-critics", action="store_true")
    parser.add_argument("--out", default="ckpt_rhododendron_v41")
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--resume", default=None, help="trainer.pt directory to resume")
    return parser.parse_args()


def main():
    args = parse_args()
    from transformers import AutoTokenizer

    from rbuild import (
        RBuildAdaptedModel,
        RBuildBaseTrainer,
        V41Config,
        continued_pretraining_batches,
        supervised_fine_tuning_batches,
    )

    config = V41Config(memory_enabled=not args.no_memory, critic_enabled=not args.no_critics)
    config.train.batch_size = args.batch_size
    config.train.grad_accum = args.grad_accum
    config.train.max_steps = args.max_steps
    config.train.warmup_steps = args.warmup_steps
    config.train.cooldown_frac = args.cooldown_frac
    config.train.lr = args.lr
    config.train.adamw_lr = args.adamw_lr
    config.train.weight_decay = args.weight_decay
    config.train.precision = args.precision
    config.validate()

    print(f"Rhododendron V4.1 | mode={args.mode} | base={args.base_model}")
    print("The base is modifiable; no architecture or checkpoint is global to R-Build.")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = RBuildAdaptedModel.from_pretrained(
        args.base_model,
        config,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    model.freeze_base(args.freeze_base)
    rows = HFRows(args)
    if args.mode == "cpt":
        batches = continued_pretraining_batches(
            tokenizer, rows, seq_len=args.seq_len, batch_size=args.batch_size
        )
    else:
        batches = supervised_fine_tuning_batches(
            tokenizer, rows, seq_len=args.seq_len, batch_size=args.batch_size
        )

    trainer = RBuildBaseTrainer(model, config)
    if args.resume:
        trainer.load_training_state(args.resume)
    trainer.fit(
        batches,
        max_steps=args.max_steps,
        save_every=args.save_every,
        save_dir=args.out,
    )
    trainer.save_checkpoint(args.out)
    tokenizer.save_pretrained(f"{args.out}/base")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

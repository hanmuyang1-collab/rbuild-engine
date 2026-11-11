#!/usr/bin/env python3
"""Run a caller-configured R-Build V4.2 scratch-ladder stage.

Ladder JSON contains exactly five stages and an RBuildConfig for every stage
that will be trained. For stage N, the data file is ``DATA_DIR/NAME.jsonl``.
The first stage initializes from scratch. Later stages strictly inherit matching
weights; shape changes require the Python API's explicit ``growth_fn``.

Local/reference example:
  python examples/train_scratch_v42.py --ladder ladder.json --data-dir data --stage A

Inspect an open production plan without allocating a model:
  python examples/train_scratch_v42.py --ladder production.json --dry-run

Stage P needs an explicit TP/EP/CP sharding wrapper and release evaluator. Use
``V42ScratchTrainer`` from Python for that integration; this portable entrypoint
intentionally refuses to pretend that DDP alone can train the 820B configuration.
"""

from __future__ import annotations

import argparse
import os

from rbuild import V42DistributedConfig, V42Ladder, V42ScratchTrainer, manifest_batches


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ladder", required=True, help="five-stage V4.2 JSON")
    parser.add_argument("--data-dir", help="directory containing STAGE.jsonl files")
    parser.add_argument("--stage", default="0", help="stage name or zero-based index")
    parser.add_argument("--out", default="ckpt_rbuild_v42")
    parser.add_argument("--resume", default=None, help="same-stage V4.2 checkpoint")
    parser.add_argument("--previous", default=None, help="previous-stage checkpoint")
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    ladder = V42Ladder.load(args.ladder)
    print(ladder.report())
    if args.dry_run:
        if ladder.production:
            plan = V42DistributedConfig()
            print("Stage-P ledger:", plan.memory_ledger().to_dict())
            print("Example launch:", " ".join(plan.torchrun_command(
                "examples/train_scratch_v42.py", 0, "MASTER_ADDR",
                ["--ladder", args.ladder, "--data-dir", "DATA_DIR", "--stage", "P"],
            )))
        return
    if ladder.production:
        raise RuntimeError(
            "The portable example does not fabricate an 820B sharding runtime. "
            "Use V42ScratchTrainer with wrap_fn and release_evaluator."
        )
    if not args.data_dir:
        raise ValueError("--data-dir is required unless --dry-run is set")
    try:
        stage_ref = int(args.stage)
    except ValueError:
        stage_ref = args.stage
    index = ladder.index(stage_ref)
    stage = ladder.stages[index]
    data_path = os.path.join(args.data_dir, f"{stage.name}.jsonl")
    batches = manifest_batches(data_path, stage.config)
    runner = V42ScratchTrainer(ladder, log_fn=print)
    result = runner.run_stage(
        index, batches, args.out, resume_checkpoint=args.resume,
        previous_checkpoint=args.previous, save_every=args.save_every,
    )
    print(f"saved -> {result['checkpoint']}")


if __name__ == "__main__":
    main()

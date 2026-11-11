#!/usr/bin/env python3
"""Validate and exec a caller-defined V4.2 multi-node torchrun launch.

Example on node 0 (run with the node-specific rank on every node):
  python examples/launch_v42.py --plan plan.json --node-rank 0 \
      --master-addr 10.0.0.10 --script examples/train_scratch_v42.py \
      --print-only -- --ladder ladder.json --data-dir /data --stage A

The plan JSON is written by your script with ``V42DistributedConfig.save``. No
hardware or weight preset is built in. The launcher configures process-group
defaults only; model sharding remains the explicit ``wrap_fn`` responsibility
of the training integration.
"""

from __future__ import annotations

import argparse
import os

from rbuild import V42DistributedConfig


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="caller-defined distributed JSON")
    parser.add_argument("--total-params", type=int, default=None,
                        help="optional model size for a memory estimate")
    parser.add_argument("--node-rank", type=int, required=True)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--script", required=True)
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main():
    args = parse_args()
    script_args = list(args.script_args)
    if script_args[:1] == ["--"]:
        script_args = script_args[1:]
    plan = V42DistributedConfig.load(args.plan)
    command = plan.torchrun_command(
        args.script, args.node_rank, args.master_addr, script_args, args.master_port
    )
    environment = os.environ.copy()
    for key, value in plan.nccl_environment().items():
        environment.setdefault(key, value)
    print("Distributed layout:", plan.to_dict())
    if args.total_params is not None:
        print("Memory ledger:", plan.memory_ledger(args.total_params).to_dict())
    print("Launch:", " ".join(command))
    if not args.print_only:
        os.execvpe(command[0], command, environment)


if __name__ == "__main__":
    main()

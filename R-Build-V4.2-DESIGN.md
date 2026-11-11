# R-Build V4.2 Design

## Status and scope

V4.2 provides interactive, caller-configured from-scratch training beside the complete V4.1 adapter/CPT/SFT route. The routes share the existing engine; neither replaces the other.

V4.2 has **no built-in weight, ladder, hardware, or data-mix presets**. Your script defines the stage list, every `RBuildConfig`, the distributed layout, and every data source. The engine validates and executes that configuration rather than imposing a target model.

The release implements strict checkpoint/resume semantics, concurrent streaming with admission filters, generic memory/cost ledgers, launch helpers, and release gates. It does not claim that any particular model has been trained or that a generic launcher is itself a sharding implementation. A production final stage requires an explicit model wrapper and measured release evidence.

V4.2 adds no modules or parameters to `RBuildModel`, `RBuildAdaptedModel`, or `ROutGenModel`. Existing forward/generation paths and parameter counters remain unchanged.

## Two routes, one engine

### V4.1: pretrained continuation

`RBuildAdaptedModel` remains a non-destructive wrapper around a caller-selected causal LM. Base `forward()` and `generate()` remain exact delegations. CPT, SFT, frozen-base adaptation, and CUA training continue through `RBuildBaseTrainer`. `v41_release_gate()` remains mandatory.

### V4.2: progressive from-scratch training

`V42Ladder` accepts any nonempty list of uniquely named stages. Each trainable stage has an explicit `RBuildConfig`; an open stage may instead carry caller-declared total/active counts for planning. `V42ScratchTrainer` constructs a native model for the first stage and advances subsequent stages through checkpoints.

```python
from rbuild import V42Ladder, V42ScratchTrainer, V42Stage

# cfg_a and cfg_b are created and modified in your script.
ladder = V42Ladder([
    V42Stage("my-first-stage", cfg_a),
    V42Stage("my-second-stage", cfg_b),
])
runner = V42ScratchTrainer(ladder)
```

Adjacent stages with identical state-dict geometry use a strict full-state transfer. If any key or shape differs, training stops and requires an explicit growth function:

```python
def grow(previous_model, current_model):
    # Map old tensors and initialize only newly created regions.
    ...
    return {"mode": "my-progressive-expansion"}

runner = V42ScratchTrainer(ladder, growth_fn=grow)
```

There is no silent fallback to random reinitialization. Same-stage resume restores model weights, fast-weight state, optimizer state, and the exact scheduler step through `v42_trainer.pt`.

## Caller-defined distributed plan

All structural `V42DistributedConfig` dimensions are required constructor arguments:

- nodes, GPUs per node, and VRAM per GPU;
- tensor parallelism;
- expert count, expert parallelism, replicas, and placements per node;
- maximum context.

Validation checks that TP groups fit inside a node, EP equals experts × replicas, and per-node placements cover EP exactly. Expert/replica placement is generic round-robin placement across the supplied nodes. At 10M context or beyond, FA3, full activation recomputation, and context parallelism/sharding are mandatory. KV and fast-weight state use GPU/CPU/NVMe tiers.

Save your script's plan for the launch helper:

```python
from rbuild import V42DistributedConfig

plan = V42DistributedConfig(
    nodes=2,
    gpus_per_node=4,
    gpu_vram_gb=80,
    tensor_parallel=2,
    n_experts=4,
    expert_parallel=8,
    expert_replicas=2,
    expert_placements_per_node=4,
    max_context=4096,
    interconnect="your-fabric",
    intranode_interconnect="your-link",
)
plan.save("plan.json")
```

`examples/launch_v42.py` requires this plan and emits per-node `torchrun` commands. Fabric-specific NCCL tuning and the actual sharded model wrapper remain deployment responsibilities.

## Memory ledger

`plan.memory_ledger(total_params)` requires the caller's model size. The default first-order training-residency calculation uses 16 bytes per parameter:

| Resident state | Bytes/parameter |
|---|---:|
| FP32 master weights | 4 |
| Adam first and second moments | 8 |
| BF16 weights | 2 |
| Gradients | 2 |

The ledger reports aggregate training residency, ideal per-GPU residency, and the remaining activation/KV/fast-weight/draft/CUA-replay budget. Real deployments must also subtract fragmentation and communication buffers.

## Data pipeline

Construct `V42DataSource` objects in your script; there is no default dataset list or mixture. Each source defines its dataset, weight, optional config/split/column, license decision, PII policy, and admission filter.

Production streaming stays closed until every source has an affirmative license decision and a row-level admission filter. `V42StreamingData` applies admission during ingestion, before tokenization or batching. Hugging Face datasets always use `streaming=True`, and bounded per-source queues are mixed concurrently by caller-assigned weight.

For native training, `V42StreamingData.batches(cfg)` reuses the existing byte-level causal batch builder. V4.1 CPT/SFT tokenization and label masking are untouched.

## Release gates

`v42_release_gate()` calls `v41_release_gate()` and additionally validates the supplied ladder and distributed invariants. Exact disabled-path output and non-regressing capability canaries remain required.

Setting `V42Ladder(production=True)` is an opt-in runtime gate, not a model preset. The final caller-defined stage then requires a release evaluator and explicit sharding wrapper before training starts.

Contract tests are not evidence of real model quality, operator competence, throughput, or training completion. Those require trained checkpoints and workload/hardware-specific measurements.

## Cost reporting

`estimate_v42_cost()` always returns **naive** and **optimized** columns. It requires explicit total tokens, total parameters, active parameters, distributed plan, GPU prices, achieved TFLOPs, and efficiency assumptions.

The first-order calculation uses dense `6 × total_params` and `6 × active_params` FLOPs/token. Either throughput profile can override these with measured end-to-end FLOPs/token to include shared-layer compute, recomputation, communication, data stalls, and checkpointing.

## Use

```python
ladder = V42Ladder.load("ladder.json")
print(ladder.report())
runner = V42ScratchTrainer(ladder, growth_fn=None)  # strict same-geometry transfer
runner.run_stage(0, stage_a_batches, "ckpt_v42")
runner.run_stage(1, stage_b_batches, "ckpt_v42")
```

Portable local/reference entrypoint:

```bash
python examples/train_scratch_v42.py \
    --ladder ladder.json --data-dir data --stage my-first-stage --out ckpt_v42
```

Inspect caller-defined plans without allocating a model:

```bash
python examples/train_scratch_v42.py --ladder ladder.json --dry-run
python examples/launch_v42.py --plan plan.json --total-params 1000000 \
    --node-rank 0 --master-addr 10.0.0.10 \
    --script examples/train_scratch_v42.py --print-only -- \
    --ladder ladder.json --data-dir /data --stage my-first-stage
```

## Verification

```bash
PYTHONPATH=. python3 tests/v42_test.py
PYTHONPATH=. python3 tests/v41_test.py
PYTHONPATH=. python3 tests/v4_test.py
PYTHONPATH=. python3 tests/cua_test.py
PYTHONPATH=. python3 tests/cua_expansion_test.py
PYTHONPATH=. python3 tests/smoke_test.py
```

# R-Build V4.2 Design

## Status and scope

V4.2 restores native from-scratch training as a five-stage progressive ladder while keeping the complete V4.1 adapter/CPT/SFT route. The routes share the existing engine; neither replaces the other.

This release implements the ladder, strict checkpoint/resume semantics, data admission and concurrent streaming, the Stage-P distributed plan, memory/cost ledgers, launch helpers, and release gates. It does **not** claim that an 820B model has been trained or that a generic launcher is itself an 820B sharding implementation. Stage P requires an explicit TP/EP/CP model wrapper and measured release evidence.

V4.2 adds no modules or parameters to `RBuildModel`, `RBuildAdaptedModel`, or `ROutGenModel`. Existing forward/generation paths and parameter counters remain unchanged.

## Two routes, one engine

### V4.1: pretrained continuation

`RBuildAdaptedModel` remains a non-destructive wrapper around a caller-selected causal LM. Base `forward()` and `generate()` remain exact delegations. CPT, SFT, frozen-base adaptation, and CUA training continue through `RBuildBaseTrainer`. `v41_release_gate()` remains mandatory.

### V4.2: progressive from-scratch training

`V42Ladder` contains exactly five named stages. Each trainable stage has an explicit `RBuildConfig`. `V42ScratchTrainer` constructs a new native model for Stage 0 and advances subsequent stages through checkpoints.

Adjacent stages with identical state-dict geometry use a strict full-state transfer. If any key or shape differs, training stops and requires:

```python
def grow(previous_model, current_model):
    # Explicitly map old tensors and initialize only newly created regions.
    ...
    return {"mode": "my-progressive-expansion"}

runner = V42ScratchTrainer(ladder, growth_fn=grow)
```

There is no silent fallback to random reinitialization. Same-stage resume restores model weights, fast-weight state, optimizer state, and the exact scheduler step through `v42_trainer.pt`.

## Locked Stage-P ledger

| Item | Locked value |
|---|---:|
| Routed experts | 16 |
| Parameters per expert | 41B |
| Shared parameters | 164B |
| Total parameters | 820B |
| Routing | Top-2 |
| Active routed-expert parameters | 82B |

The shared allocation covers attention, embeddings, routers, V3 critics, draft-verify, and fast-weight machinery. The 82B invariant refers to active routed-expert parameters. For end-to-end compute estimates, callers should supply measured FLOPs/token that also account for shared layers, communication, and activation recomputation.

The four intermediate stages remain open. `V42Ladder.production_ladder()` accepts four caller-defined stages and appends the locked Stage P; open stages serialize as `null` rather than receiving invented sizes. Training validation rejects open configurations.

## Distributed plan

The production plan is 16 nodes × 8 H200 141GB GPUs = 128 GPUs and 18,048GB aggregate VRAM, connected by NDR 800G InfiniBand.

- **TP=4:** tensor-parallel groups fit within each eight-GPU node, keeping attention, critics, and fast-weight operations on NVLink rather than InfiniBand.
- **EP=32:** two replicas of each of 16 experts. Every node holds two expert placements: one primary and one replica. Expert `e` is primary on node `e` and replicated on node `(e + 8) mod 16`; routing must be replica-aware.
- **10M context:** FA3, full activation recomputation, and context parallelism/sharding are mandatory. KV and fast-weight state use GPU/CPU/NVMe tiers.

`V42DistributedConfig` validates these invariants and emits per-node `torchrun` commands. `examples/launch_v42.py` validates and executes the launch. Fabric-specific NCCL tuning and the actual sharded model wrapper remain explicit deployment responsibilities; portable async-error-handling defaults do not pretend to replace NCCL tests.

## Memory ledger

The first-order training-residency calculation uses 16 bytes per parameter:

| Resident state | Bytes/parameter |
|---|---:|
| FP32 master weights | 4 |
| Adam first and second moments | 8 |
| BF16 weights | 2 |
| Gradients | 2 |

At 820B this is 13,120GB across the cluster, or 102.5GB/GPU under ideal sharding. The remaining 4,928GB aggregate is the budget for activations, KV, fast weights, the draft head, and CUA replay. It is not spare weight capacity, and real deployments must subtract fragmentation and communication buffers.

## Data pipeline

`default_v42_data_sources()` defines four weighted inputs:

1. `HuggingFaceFW/fineweb-edu` — base knowledge.
2. `allenai/dolma` — multi-source anchor.
3. Exactly one of `HuggingFaceTB/finemath` or `LLM360/MegaMath` — mathematics, avoiding double-counted math tokens.
4. `bigcode/the-stack-v2` — code and later CUA grounding.

Dataset identifiers are not license approvals. Production streaming stays closed until every source has an affirmative license decision and a row-level admission filter. `V42StreamingData` applies admission during ingestion, before tokenization or batching; license and PII handling are not post-hoc cleanup. Hugging Face datasets always use `streaming=True`, and bounded per-source queues are mixed concurrently by weight.

For native training, `V42StreamingData.batches(cfg)` reuses the existing byte-level causal batch builder. V4.1 CPT/SFT tokenization and label masking are untouched.

## Release gates

`v42_release_gate()` calls `v41_release_gate()` and additionally validates ladder and distributed invariants. Exact disabled-path output and non-regressing capability canaries remain required. A Stage-P run also requires a caller-supplied release evaluator and an explicit sharding wrapper before training starts.

Contract tests are not evidence of real model quality, operator competence, throughput, or training completion. Those require trained checkpoints and workload/hardware-specific measurements.

## Cost reporting

`estimate_stage_p_cost()` always returns **naive** and **optimized** columns. It requires explicit token count, GPU price, achieved TFLOPs, and efficiency assumptions. No grant, cloud discount, or committed H200 allocation is assumed.

The default first-order calculation compares dense `6 × 820B` with `6 × 82B` FLOPs/token. Production estimates should override both with measured end-to-end FLOPs/token because the simplified active-parameter figure omits shared-layer compute, recompute, communication, data stalls, and checkpointing.

## Use

```python
from rbuild import V42Ladder, V42ScratchTrainer

ladder = V42Ladder.load("ladder.json")
print(ladder.report())
runner = V42ScratchTrainer(ladder, growth_fn=None)  # strict same-geometry transfer
runner.run_stage(0, stage_a_batches, "ckpt_v42")
runner.run_stage(1, stage_b_batches, "ckpt_v42")
```

Portable local/reference entrypoint:

```bash
python examples/train_scratch_v42.py \
    --ladder ladder.json --data-dir data --stage A --out ckpt_v42
```

Inspect a production plan without allocating a model:

```bash
python examples/train_scratch_v42.py --ladder production.json --dry-run
python examples/launch_v42.py --node-rank 0 --master-addr 10.0.0.10 \
    --script examples/train_scratch_v42.py --print-only -- \
    --ladder production.json --data-dir /data --stage P
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

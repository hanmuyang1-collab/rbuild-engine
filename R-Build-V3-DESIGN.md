# R-Build v3 — Design Blueprint

**Status:** implemented on the `v3` branch, counter-verified.
**Lineage:** extends v2.1 (cache-loop + parallel-bundle skeleton) without
breaking it — every v3 system can be disabled independently, restoring the
exact v2 architecture and parameter count.

---

## 0. Thesis

v2 could *remember* without training (gradient-free fast-weight writes).
v3 makes the model **self-governing**: it decides how much compute each
token needs, judges its own work with dedicated critic hardware, and learns
from what it observes — *while it runs*, not in a separate training phase.

Three organs make this work:

| organ | where | job |
|---|---|---|
| **Critic experts** (X per panel) | extraction loop + every generative stage | score completeness; gate halting; verify notes |
| **ACT halting** | extraction loop | run until Y critics are satisfied, pay a ponder cost |
| **Noting experts** | final hidden states | propose facts; verified ones are learned immediately |

Design rule carried from v2 and strengthened: **critics get more capacity
than the working experts they judge** (`critic_capacity_mult`, default 2×).
Verification is cheaper than generation, but it must never be the
bottleneck of trust.

---

## 1. Critic-gated adaptive extraction (ACT-style halting)

The v2 cache loop ran a fixed `n_loops`. In v3 it is the **extraction
loop** with adaptive depth:

```
state x enters the loop
for iteration l = 1 .. max_loops:
    x <- line(x)                      # shared-weight blocks (MoD-routed)
    x <- pull_cache(x)                # delta-rule read, gated injection
    p_l <- CriticPanel(x)             # X critics vote per token
    per token: halting mass += p_l    # ACT accumulation
    token halts when mass >= 1 - eps  # remainder closes mass to 1
    loop exits when >= Y critics satisfied on average (or max_loops)
halted tokens keep their final state; running tokens continue
```

- **Per-token halting** is ACT-style: cumulative satisfaction is the
  halting probability mass; each token's total mass is exactly
  1 + remainder.
- **Ponder cost** = mean(loops_taken + remainders), weighted by
  `critic.halt_loss_weight` and added to the training loss. The model
  literally pays for thinking longer.
- **Global early exit** implements the blueprint rule directly: *the
  extraction loop runs until Y critic experts are satisfied.*
- `min_loops` guarantees a floor; `max_loops` is the cap (set to the
  ladder's `n_loops` in presets).

With `critic.enabled=False` the loop runs exactly `n_loops` iterations —
bit-identical v2.

## 2. Critic experts on the generative layers

Every parallel bundle stage (the generative layers) carries its own
`CriticPanel` of X critics beside the working MoE branches:

- Critic hidden width = `expert_ffn_dim × critic_capacity_mult` — **more
  capacity than the working experts**, per the blueprint.
- Stage critics score the stage's hidden state (`stage.critic_verdict(x)`
  → satisfaction, n_satisfied).
- Their primary duty is **verification**: the self-training pipeline only
  trusts a note when the final stage's critics approve it (§3).

Critics cost ~0.5% of total parameters at s1 (101M of 20.0B).

## 3. Noting experts + non-separate self-training

"Non-separate" = there is **no separate training phase and no separate
verification phase**. One forward pass does all of it:

```
forward pass (training OR generation — running is running)
  └► final hidden states
       ├► NotingExperts (separate note-takers) ─► candidate (key, value, confidence)
       ├► stage critics verify: keep if
       │     n_satisfied >= verify_y_critics
       │     AND mean satisfaction >= verify_threshold
       │     AND note confidence >= min_confidence
       ├► verified ─► fast-weight memory write (delta rule, gradient-free,
       │              ~zero extra RAM — the model has *already learned* it)
       └► verified ─► VerifiedNoteBuffer (CPU, fp16, capped — least RAM)
                        └► Trainer.self_train_step(): replay the noted
                           hidden states; train the native fact projections
                           (fact_key_proj / fact_value_proj — the same path
                           remember() uses) to reproduce the verified
                           (key, value) pairs. Score-weighted MSE, own LR,
                           WSD schedule untouched.
```

- **Fast path (parallel running and learning):** gradient-free delta-rule
  writes happen inside the observing forward pass itself.
- **Slow path (consolidation):** batched, rare, GPU-efficient; runs on
  demand or every `train.self_train_every` steps inside `fit()`.
- **RAM discipline:** notes live on CPU in fp16 (`buffer_capacity` cap,
  highest-scoring kept); nothing touches the GPU until consolidation.
- Untrained critics start neutral (satisfaction ≈ 0.5), so the default
  threshold 0.6 **rejects everything until the critics learn to judge** —
  the gate fails closed, not open.

Observability: `model.self_learn_stats()` → verified / rejected /
accept-rate / buffered; checkpoint `meta.json` carries the same.

## 4. Thinking-mode creator

Thinking modes are named, user-creatable runtime presets that retune the
adaptive machinery without rebuilding the model:

```python
model.thinking_mode.deep()        # built-in
model.thinking_mode.deep(12)      # positional value = max_loops override
model.thinking_mode.create("exam", max_loops=10, y_critics=3,
                           halt_threshold=0.9, temperature=0.2)
model.thinking_mode.exam()        # user modes are first-class
```

Mode knobs: `max_loops`, `y_critics`, `halt_threshold`, `temperature`,
`top_p`, `self_observe`.
Built-ins: `fast` (2 loops, no observation) · `balanced` · `deep` ·
`careful` · `research` (24 loops, strictest gate).
Custom modes persist into checkpoints via `thinking.custom_modes`.

## 5. Vision: ViT, encoderless, VaWU

| mode | what happens |
|---|---|
| `vit` (v2.1) | ViT tower encodes patches; projected to d_model |
| `encoderless` (v3) | **no vision encoder at all** — patches are normalized + linearly projected straight into d_model; the LLM's own stages do the seeing |

`vision.vawu=True` (**Video-as-Whole-Understanding**): a learned-query
attention pooler compresses all frames into `vawu_tokens` whole-video
summary tokens, *prepended* to the frame stream — the model reads the video
as a whole before its parts. Works with both modes.

Blind default unchanged: `vision.enabled=False` builds nothing.

## 6. Native actuation

Tools are snails; v3 makes the model click **by generating a token**.

- `actuation.enabled=True` extends the output space with action tokens:
  `wait` · `click` on a `screen_grid²` grid · `scroll` (±steps) ·
  `type_begin` / `type_end`.
- A dedicated `ActuationHead` produces the action columns of the logits,
  so the action space stays clean even with tied embeddings. One forward
  step = one action; no parsing harness.
- `model.action_codec.decode_actions(tokens)` → `Action` objects with
  `.execute(driver)` (pyautogui-like).
- Grounding pairs with encoderless vision + VaWU: the screen is just
  frames, frames are just tokens, and the same forward that sees the
  screen also clicks it.

## 7. Counter verification (project invariant)

`cfg.count_parameters()` mirrors the v3 module tree **exactly** — every
critic panel, note-taker, action head, encoderless/ViT/VaWU vision path,
and the actuation-extended vocabulary. The invariant

```python
assert sum(p.numel() for p in RBuildModel(cfg).parameters())
       == cfg.count_parameters()["total_params"]
```

holds for every configuration, and `report()` always shows
naive-vs-optimized cost side by side. v3 overhead at s1: +101M critics,
+19M noting (≈0.6% of 20B total).

## 8. Backward compatibility

- `critic.enabled=False`, `noting.enabled=False`, `actuation.enabled=False`,
  `vision.enabled=False` → exact v2 architecture and parameter count.
- v2 checkpoints load: `from_dict` skips unknown/missing sections; new
  sections take defaults.
- The fast-weight memory matrix, checkpoint format, and
  `Trainer` save/load are unchanged.

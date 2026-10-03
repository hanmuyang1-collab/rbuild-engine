# R-Build v3

**A fully user-modifiable, open-source LLM architecture + training-speed engine — now self-governing.**
Every knob is yours: programmatically, through an interactive widget panel in
Colab/Jupyter, or a console prompt anywhere else.

## What's new in v3

```
tokens ─► embed ─► EXTRACTION LOOP ─► PARALLEL STAGE 1 ─► ... ─► PARALLEL STAGE N ─► head
                    │  single line of    │  n_branches working experts per stage
                    │  layers, ACT-      │  + X CRITIC EXPERTS per stage
                    │  halted: runs      │  (more capacity than the workers —
                    │  until Y critics   │  they verify the bundle and gate
                    │  are satisfied,    │  the self-training pipeline)
                    │  pulling the       │
                    │  fast-weight       ▼
                    │  cache each loop   logits (+ native action tokens)
                    ▼
            FastWeightMemory — delta-rule key→value matrix.
            v3: critic-verified notes written by the model's own
            noting experts — it learns WHILE running, no separate phase.
```

1. **Critic-gated adaptive extraction (ACT-style halting).** The old fixed
   cache loop is now an *extraction loop*: after each iteration, X parallel
   critic experts score every token's hidden state. Tokens halt as their
   cumulative satisfaction crosses 1 (ACT); the loop early-exits once Y
   critics are satisfied on average. An ACT ponder cost (loops taken +
   remainders) joins the training loss, so the model learns to spend only
   the compute each token needs.
2. **Critic experts on the generative layers.** Every parallel stage carries
   its own panel of X critics, each with `critic_capacity_mult`× the width of
   a working expert — verification gets more capacity than generation.
3. **Noting experts + critic-verified non-separate self-training.** Separate
   note-taker experts watch the final hidden states during normal use
   (generation included) and propose facts. Critics verify each note;
   verified notes are written into the fast-weight memory *gradient-free,
   immediately* (the model learns while running, ~zero extra RAM) and queued
   in a CPU fp16 buffer for `Trainer.self_train_step()` to consolidate into
   the slow weights. Running and learning are the same pass.
4. **Thinking-mode creator.** `model.thinking_mode.<mode>(<value>)` retunes
   loops, Y-critics, thresholds and sampling live — built-ins
   `fast/balanced/deep/careful/research`, and mint your own with
   `model.thinking_mode.create("exam", max_loops=10, y_critics=3)`.
5. **VL & VaWU without a vision encoder.** `vision.mode="encoderless"`:
   patches are normalized and projected straight into `d_model` — the LLM
   itself is the vision encoder. `vision.vawu=True` adds
   Video-as-Whole-Understanding: learned-query attention pooling compresses
   all frames into whole-video summary tokens, prepended to the frame
   stream. (The v2.1 ViT tower remains as `vision.mode="vit"`.)
6. **Native actuation.** `actuation.enabled=True` reserves action tokens in
   the output space — click(x, y) on a screen grid, scroll, type, wait —
   produced by a dedicated action head. Generating a token *is* the action;
   no external tool loop.

Everything v3 can be switched off independently; with `critic`, `noting`,
`actuation` and `vision` all disabled the architecture is bit-identical to
v2, and the parameter counter still matches the built model exactly.

## Install

```bash
pip install git+https://github.com/hanmuyang1-collab/rbuild-engine@v3   # v3 branch
pip install .                       # or from a local clone
pip install .[notebook]             # + ipywidgets for the interactive panel
pip install .[train]                # + transformers/datasets for the training script
```

## Use it — everything is modifiable

```python
from rbuild import RBuildConfig, RBuildModel, Trainer, preset

cfg = preset("s1")                  # counter-verified: 20.0B total / 3.5B active
cfg.critic.n_critics = 8            # X critics per panel — change anything
cfg.critic.y_critics = 4            # Y must be satisfied to halt extraction
cfg.train.precision = "bf16"
print(cfg.report())                 # params + naive-vs-optimized cost, always both

model = RBuildModel(cfg)
assert sum(p.numel() for p in model.parameters()) == cfg.count_parameters()["total_params"]
```

### Thinking modes

```python
model.thinking_mode.deep()                    # built-in: longer extraction, stricter critics
model.thinking_mode.deep(12)                  # positional value = max_loops override
model.thinking_mode.create("exam", max_loops=10, y_critics=3,
                           halt_threshold=0.9, temperature=0.2)
model.thinking_mode.exam()                    # your mode is now native
model.thinking_mode.list()                    # all modes + knobs
```

### Self-training while running

```python
model.eval()
out = model.generate(ids)           # noting experts observe this pass
model.self_learn_stats()            # {'notes_verified': ..., 'accept_rate': ..., 'buffered': ...}

trainer = Trainer(model, cfg)
trainer.self_train_step()           # consolidate verified notes into slow weights
# or automatically during fit: cfg.train.self_train_every = 100
```

### Native actuation

```python
cfg.actuation.enabled = True
cfg.actuation.screen_grid = 64      # 64x64 click grid
model = RBuildModel(cfg)
out = model.generate(ids)
for action in model.action_codec.decode_actions(out[0]):
    ...                             # Action(click, x=0.31, y=0.81), Action(scroll, dy=-2), ...
```

### Vision: ViT, encoderless, VaWU

```python
cfg.vision.enabled = True
cfg.vision.mode = "encoderless"     # v3: no vision encoder at all
cfg.vision.vawu = True              # whole-video summary tokens
cfg.vision.image_token_id = 128001  # reserved placeholder id
model = RBuildModel(cfg)
logits, loss = model(ids, targets=ids, images=imgs)   # imgs: (B, frames, C, H, W)
```

### Interactive panel (Colab / Jupyter)

```python
from rbuild import interactive
ui = interactive.launch()           # widgets for every value, v3 sections included
model = ui.model
interactive.chat(model, encode, decode)
# /mode deep      switch thinking mode live      /modes     list modes
# /selfstats      self-learning counters         /remember  gradient-free fact write
```

## Stage ladder presets (continued-training path)

| preset | total (v3) | active/token | v3 additions |
|---|---|---|---|
| s1 | 20.0B | 3.5B | +101M critics, +19M noting |
| s2 | 90.7B | 9.0B | critics scale with stage widths |
| s3 | 118.4B | 19.4B | wider-per-token, not deeper |
| s4 | 219.3B | 40.9B | |
| s5 | 415.0B | 61.9B | |

Working-expert sizes are the v2 ladder; v3 critics + noting experts add
their counter-verified parameters on top (`report()` splits them out).
Critics cost ~0.5% of total parameters. All presets are counter-verified;
treat them as starting points and retune in the panel.

## Training scripts

- `examples/v3_quickstart.py` — every v3 feature in one tiny CPU run:
  counter verification, ACT halting, thinking modes, self-training,
  encoderless VL + VaWU, native actuation.
- `examples/train_rhododendron_lite.py` — full training run for
  **rhododendron-lite (20B-A3.5B, blind)**: HF streaming data, Muon + WSD,
  bf16/fp8 flags, gradient checkpointing, periodic checkpoints, resume,
  FSDP/DDP via torchrun, optional push to the GeoThinkAI org.

See **R-Build-V3-DESIGN.md** for the architecture blueprint.

Apache-2.0. Part of the GeoThinkAI R-Build project.

# R-Build v4.2

**A fully user-modifiable, open-source architecture + training-speed engine — with a five-stage from-scratch ladder running beside non-destructive pretrained conversion, continued pretraining, and supervised fine-tuning, while retaining exact-AR draft–verify decoding, self-governing, and self-serving.**
Every knob is yours: programmatically, through an interactive widget panel in
Colab/Jupyter, or a console prompt anywhere else.

v3.1 merges **R-Run** into this repo — R-Build builds and trains the models,
R-Run serves them. One repo, one engine: `rrun serve` / `rrun swap`.

## What's new in v4.2

V4.2 restores **from-scratch training without removing V4.1**. `V42Ladder` holds
exactly five progressive stages; `V42ScratchTrainer` initializes the first stage,
strictly transfers matching weights between stages, requires an explicit
`growth_fn` for geometry changes, and resumes model, fast-weight, optimizer, and
scheduler-step state. No stage is silently reinitialized.

The production ladder ends at the locked **Stage P: 16 × 41B experts + 164B
shared = 820B total; Top-2 routing activates 82B expert parameters**. The four
intermediate stage sizes remain caller-defined and serialize as open until
configured. The distributed plan validates 128× H200 (16 nodes × 8), node-local
TP=4, EP=32 through two replicas per expert, replica-aware placement, and the
FA3 + full-recompute + context-sharding contract for 10M context.

```python
from rbuild import V42Ladder, V42ScratchTrainer

ladder = V42Ladder.load("ladder.json")       # five configured stages
print(ladder.report())
runner = V42ScratchTrainer(ladder)           # strict same-geometry transfer
runner.run_stage(0, stage_a_batches, "ckpt_v42")
runner.run_stage(1, stage_b_batches, "ckpt_v42")
```

`V42StreamingData` concurrently mixes FineWeb-Edu, Dolma, exactly one of
FineMath/MegaMath, and The Stack v2 using bounded Hugging Face streams.
Production ingestion requires explicit license approval and row-level admission
filters before tokenization. `estimate_stage_p_cost()` always returns naive and
optimized columns from explicit throughput and price assumptions.

Stage P is an implementation plan, not a claim of completed 820B training. It
requires an explicit TP/EP/CP `wrap_fn` and release evaluator; the portable
example refuses to substitute ordinary DDP. V4.2 adds no model modules or
parameters, and `v42_release_gate()` reuses V4.1's exact-output and capability
preservation checks.

```bash
python examples/train_scratch_v42.py --ladder ladder.json --data-dir data --stage A
python examples/train_scratch_v42.py --ladder production.json --dry-run
PYTHONPATH=. python3 tests/v42_test.py
```

See **R-Build-V4.2-DESIGN.md** for the memory ledger, expert placement, data
admission contract, launch helper, cost assumptions, and production boundaries.

## What's new in v4.1

V4.1 makes pretrained continuation the default build path instead of requiring a
from-scratch run. `RBuildAdaptedModel` wraps **any PyTorch/Hugging Face causal-LM
architecture** without replacing its native layers: base forward and generation
remain exact delegations, while R-Build critic panels and delta-rule fast-weight
memory are additive. The original native R-Build architecture, MoD/MoE/shared
experts, V4 draft–verify decoder, media heads, actuation, watermarking, and R-Run
remain available — no feature is removed.

Both training modes reuse the original Muon + WSD stack:

- **Continued pretraining (CPT):** packed raw-document causal-LM batches.
- **Supervised fine-tuning (SFT):** prompt/response batches with prompt labels
  masked to `-100`, so loss is computed on responses rather than instructions.
- Full-base or adapter-only training, gradient accumulation, bf16/fp8 contract,
  periodic checkpoints, resume state, and capability-preservation release gates.

The base checkpoint is always caller-selected and modifiable. No project-specific
or global checkpoint default is hardcoded.

**Every R-Build model is a CUA.** Native text models, R-OutGen media models, and
adapted pretrained models all expose `model.cua` with the same reserved action
codec and dedicated action head. Use `cua.observe(...)`/`cua.propose(hidden)` to
propose click/scroll/type/wait actions, `cua.action_loss(...)` for supervised CUA
trajectories, and `cua.execute(...)` with a real driver to perform OS operations.
The action head requires computer-use training; constructing it does not by itself
prove operator competence. Irreversible action batches require `confirmed=True`,
and confirmation is checked before any side effect.

The complete CUA reliability and adaptation stack is implemented on the existing
codec, action head, critics, hidden states, and Muon/WSD trainer:

1. **Uncertainty abstention:** `propose(..., max_entropy=..., min_confidence=...)`
   emits `wait` or requires confirmation per batch row.
2. **Critic-verified candidates:** `propose_verified()` ranks top-k actions with the
   existing critic panel; `act_verified()` also checks an expected postcondition.
3. **Event-driven observation:** `ScreenChangeDetector` and `observe_if_changed()`
   skip unchanged screens while retaining bounded timed polling.
4. **Shadow mode:** `ShadowDriver` records actions without calling a wrapped real
   driver, so dry runs cannot produce OS side effects.
5. **Verified trajectory replay:** `CUATrajectoryBuffer` accepts verified examples
   only, stores hidden states as bounded CPU fp16, and pads action labels with `-100`.
6. **Application policies:** `CUAPolicy`/`PolicyDriver` enforce action allowlists,
   coordinate/scroll/text limits, deny rules, confirmation, and rate limits before
   the first side effect.
7. **Release gates:** `run_cua_contract_canaries()` and `cua_release_gate()` check
   all three model classes and reject safety regressions.
8. **Deterministic audit/replay:** `CUAReplayLedger` persists bounded rotating JSONL
   records and replays the recorded actions rather than sampling new ones.
9. **Frozen-base adaptation:** `freeze_for_cua()` plus `RBuildBaseTrainer.fit_cua()`
   trains only CUA components through the existing optimizer and scheduler.
10. **Encoderless screenshots:** adapted models can opt into one convolutional patch
    projector and `observe_screenshot()` without a second vision encoder.
11. **Application adapters:** named, zero-initialized low-rank residual adapters
    specialize the shared CUA hidden path and round-trip through checkpoints.
12. **Interruption/recovery curriculum:** `build_recovery_curriculum()` derives
    verified variants for dialogs, moved scroll targets, loading, stale screens,
    and failed clicks.

Runtime verification, policies, auditing, replay, and recovery add no model
parameters. Screenshot and application-adapter weights are adapted-model-only and
opt-in; native/R-OutGen parameter counters and adapted base `forward()`/`generate()`
delegation are unchanged. Postcondition failure is audited but cannot roll back
side effects that already occurred, so destructive work still requires policy and
confirmation controls. Contract canaries validate interfaces and safety behavior;
grounded application canaries remain necessary before a real release.

```python
from rbuild import RBuildAdaptedModel, RBuildBaseTrainer, V41Config
from rbuild import continued_pretraining_batches, supervised_fine_tuning_batches

cfg = V41Config()                    # memory + critics; every value modifiable
cfg.train.max_steps = 10_000
model = RBuildAdaptedModel.from_pretrained(
    "your/selected-base", cfg, torch_dtype="auto", low_cpu_mem_usage=True)

cpt = continued_pretraining_batches(tokenizer, documents, seq_len=4096)
sft = supervised_fine_tuning_batches(tokenizer, prompt_response_rows, seq_len=4096)
trainer = RBuildBaseTrainer(model, cfg)
trainer.fit(cpt)                     # continued pretraining
trainer.fit(sft)                     # optional supervised fine-tuning
trainer.save_checkpoint("ckpt_v41")
```

Frozen-base CUA training and application specialization reuse that adapted model:

```python
from rbuild import CUATrajectory, CUATrajectoryBuffer

model.add_cua_adapter("desktop", rank=8, alpha=16.0)
model.activate_cua_adapter("desktop")
model.freeze_for_cua()               # base, memory, and critics stay frozen

replay = CUATrajectoryBuffer(max_trajectories=1024, max_bytes=2 * 1024**3)
replay.add(CUATrajectory(hidden_states, action_token_ids, verified=True))
cua_trainer = RBuildBaseTrainer(model, cfg)
cua_trainer.fit_cua(replay.batches(batch_size=4), max_steps=100)
```

A verified dry run uses the existing critics and a fake driver; `real_driver` is
never called:

```python
shadow = model.cua.shadow(real_driver)
shadow.postconditions["target_ready"] = True
decision = model.cua.act_verified(
    hidden_states, shadow, candidates=4,
    expected_postcondition="target_ready",
)
print(decision.verification["candidate_votes"], shadow.events)
```

For a caller-selected pretrained starting point:

```bash
python examples/train_base_v41.py --mode cpt \
    --base-model your/selected-base --dataset HuggingFaceFW/fineweb-edu \
    --dataset-name sample-10BT
python examples/train_base_v41.py --mode sft \
    --base-model your/selected-base --dataset your/sft-data
```

`v41_release_gate()` requires exact base/adapted output on the disabled path and
non-regressing capability canaries. Run the acceptance and smoke suites:

```bash
PYTHONPATH=. python3 tests/v4_test.py
PYTHONPATH=. python3 tests/v41_test.py
PYTHONPATH=. python3 tests/cua_test.py
PYTHONPATH=. python3 tests/cua_expansion_test.py
PYTHONPATH=. python3 tests/smoke_test.py
```

See **R-Build-V4.1-DESIGN.md** for the conversion and training contract.

## What's new in v4

R-Build-v4 adds a tiny absorbing-mask discrete-diffusion draft head with shared
input/output embeddings, masked-token CE, and main-model KL distillation
(`kl_weight=0.05`). The default decode schedule is `L=8`, `K=3`, and `R_max=2`.
The main model scores each proposal and commits only the critic-approved
longest matching prefix; a mismatch commits the authoritative AR token. Only
accepted/authoritative tokens enter the output and watermark path.

Use `model.generate(ids, use_v4=True)` to enable it explicitly. `use_v4=False`
provides the reference AR path. `v4_release_gate()` requires acceptance at least
0.60, canary quality preservation, and byte-identical AR output before
`model.freeze_v4()` will freeze the drafter. An untrained drafter does not have
a promised speedup: the 3–4× figure remains a target to measure on trained
checkpoints and real hardware.

Run the focused acceptance suite:

```bash
PYTHONPATH=. python3 tests/v4_test.py
```

The complete v3 architecture remains available below.

## What's new in v3

```
tokens ─► embed ─► EXTRACTION LOOP ─► PARALLEL STAGE 1 ─► ... ─► PARALLEL STAGE N ─► head
                    │  single line of    │  n_branches working experts per stage
                    │  layers, ACT-      │  + X CRITIC EXPERTS per stage
                    │  halted: runs      │  (more capacity than the workers —
                    │  until Y critics   │  they verify the bundle and gate
                    │  are satisfied,    │  the self-training pipeline)
                    │  pulling the       ▼
                    │  fast-weight       logits (+ native action tokens)
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
   **v3.1: this is opt-in.** NSCT — non-separate continuous training —
   is OFF by default (`noting.nsct=False`) — turn it on per config, per
   call (`model.set_nsct(True)`), per chat command (`/nsct on`), or
   per thinking mode (`self_observe`).
4. **Thinking-mode creator.** `model.thinking_mode.<mode>(<value>)` retunes
   loops, Y-critics, thresholds and sampling live — built-ins
   `instant/fast/balanced/deep/careful/research`, and mint your own with
   `model.thinking_mode.create("exam", max_loops=10, y_critics=3)`.
   Every mode defines its built-in reasoning `effort` (float, invisible)
   and a `reasoning` switch (`False` = instant, no thinking); the
   caller-facing **selectable effort** (`{effort:'low|medium|high'}` tags)
   multiplies it and genuinely scales extraction depth, critic strictness
   and sampling sharpness.
5. **VL & VaWU without a vision encoder.** `vision.mode="encoderless"`:
   patches are normalized and projected straight into `d_model` — the LLM
   itself is the vision encoder. `vision.vawu=True` adds
   Video-as-Whole-Understanding: learned-query attention pooling compresses
   all frames into whole-video summary tokens, prepended to the frame
   stream. (The v2.1 ViT tower remains as `vision.mode="vit"`.)
6. **Universal CUA / native actuation.** Every model has a dedicated CUA action
   head and `model.cua` runtime — click(x, y) on a screen grid, scroll, type,
   and wait. By default, reserved action tokens also extend native text-generation
   logits. Setting `actuation.enabled=False` preserves legacy text-logit shapes
   but does not remove the separate CUA interface. Real side effects require an
   explicit driver, and irreversible actions require confirmation.
7. **Generation watermarking.** `watermark.enabled=True` biases sampling with
   a secret-keyed green list, so anything your model writes is provably
   yours — `WatermarkDetector` runs a z-test with the same key. Pure
   sampling-time signal: zero parameters, checkpoints unaffected.
8. **R-Run serving, merged in (v3.1).** The `rrun` package hosts any model
   with one command and hot-swaps the resident model with **zero server
   restart** and a **full KV cache** for the new model — OpenAI-compatible
   API included. The native `rbuild` backend serves R-Build checkpoints
   directly (no conversion); `vllm`, `hf` and `mock` backends cover
   everything else.

Critic, noting, and vision features remain independently configurable. CUA is
universal; `actuation.enabled` controls only text-logit extension. The parameter
counter includes the dedicated action head and matches the built model exactly.

## Install

```bash
pip install git+https://github.com/hanmuyang1-collab/rbuild-engine@v4   # v4 branch
pip install .                       # or from a local clone
pip install .[notebook]             # + ipywidgets for the interactive panel
pip install .[train]                # + transformers/datasets for the training script
pip install .[serve]                # + fastapi/uvicorn for the rrun server
```

## Use it — everything is modifiable

```python
from rbuild import RBuildConfig, RBuildModel, Trainer, preset

cfg = preset("s1")                  # counter-verified: 20.6B total / 3.5B active
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
model.thinking_mode.instant()                 # reasoning=False — instant, no thinking
model.thinking_mode.create("exam", max_loops=10, y_critics=3,
                           halt_threshold=0.9, temperature=0.2,
                           effort=1.5, reasoning=True)
model.thinking_mode.exam()                    # your mode is now native
model.thinking_mode.list()                    # all modes + knobs (effort & reasoning included)
```

Every mode carries two extra definitions: **`effort`** (a float — the
mode's built-in reasoning effort, invisible to end users) and
**`reasoning`** (a bool — `False` = instant, no thinking: the extraction
loop runs its minimum and answers immediately). On top of that sits the
**selectable effort**: a built-in ladder of named multipliers — `low`
(×0.5), `medium` (×1.0), `high` (×2.0) — that callers pick *by tag*,
unlike the invisible built-in effort:

```python
"explain this paper {effort:'high'}"          # tag inside the prompt (chat)
model.generate(ids, effort="high")            # per call
model.generate_image(ids, effort="low")       # media generation too
model.thinking_mode.set_effort("high")        # or persistently

# effective effort = mode's built-in effort x the selectable multiplier
model.thinking_mode.effective_effort()        # deep (2.0) x high (2.0) = 4.0
```

And it is real, not cosmetic: raising the selectable effort deepens the
extraction loop, requires more critics to agree, tightens the halt
threshold, and sharpens sampling — more effort = measurably deeper,
higher-quality thinking; `low` answers faster and shallower. `medium`
(×1.0) is the default and reproduces each mode's declared behavior
exactly. Add your own rungs via `cfg.thinking.selectable_efforts`.

### Self-training while running (NSCT)

NSCT = non-separate continuous training: **training and running happen
simultaneously**, in the same pass, at low RAM — no separate training
phase. It is opt-in (off by default):

```python
cfg.noting.nsct = True              # OPT-IN: off by default
model = RBuildModel(cfg)
model.eval()
out = model.generate(ids)           # noting experts observe this pass
model.self_learn_stats()            # {'notes_verified': ..., 'accept_rate': ..., 'buffered': ...}

model.set_nsct(False)               # toggle at runtime — pure inference again
model.set_nsct(True)                # learn while running again

trainer = Trainer(model, cfg)
trainer.self_train_step()           # consolidate verified notes into slow weights
# or automatically during fit: cfg.train.self_train_every = 100
```

### Serve it — R-Run is built in

```bash
rrun serve ./ckpt --backend rbuild          # host a native R-Build checkpoint
rrun swap  ./ckpt_v2 --backend rbuild       # hot-swap: wipe old, full KV for new
rrun status                                 # resident model + KV cache state
rrun serve Qwen/Qwen3-32B                   # HF/vLLM models work too
```

The HTTP server never restarts on swap — same port, same connections, new
model. OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/models`)
plus admin endpoints (`/admin/swap`, `/admin/wipe`, `/admin/status`).
Byte-level tokenization by default (matches local .txt training); pass a
tokenizer for HF-trained checkpoints:

```python
from rrun.backends import RBuildBackend
from rrun.engine import RRunEngine

eng = RRunEngine(RBuildBackend(tokenizer="gpt2", watermark=True))
eng.serve("./ckpt")
eng.chat([{"role": "user", "content": "hello"}])
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

### Generation watermarking

```python
cfg.watermark.enabled = True
cfg.watermark.key = "my-secret"     # keep it private
model = RBuildModel(cfg)
out = model.generate(ids)           # quietly watermarked

from rbuild import WatermarkDetector
det = WatermarkDetector(cfg.watermark, cfg.effective_vocab_size())
det.detect(out)                     # {'z_score': 9.1, 'watermarked': True, ...}
# wrong key -> not detected; unwatermarked text -> z ≈ 0
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

### Data: manual JSON manifest + no-fuss HF image/video

Every model type (blind, ViT, encoderless) trains from one manual JSON
manifest — text plus links to pictures/videos (local paths or https URLs):

```json
[{"text": "a cat on a mat", "images": ["cat.jpg", "https://…/cat2.png"]},
 {"text": "a short clip",    "video":  "clip.mp4"},
 {"text": "text only is fine too"}]
```

```python
from rbuild import manifest_batches, hf_image_batches, hf_video_batches

batches = manifest_batches("data.json", cfg)     # .json or .jsonl, any model type
batches = hf_image_batches("lambdalabs/naruto-blip-captions", cfg)  # vision on
batches = hf_video_batches("friedrichor/MSR-VTT", cfg)              # vision on
trainer.fit(batches)   # (x, y) or (x, y, images) — Trainer handles both
```

No fuss: HF columns auto-detect from the first row (override with
`text_column=` / `image_column=` / `video_column=`), images decode/resize
themselves, videos sample frames uniformly (decord → imageio → ffmpeg,
whatever is installed), and `<image>` placeholder runs are spliced to the
exact length the vision tower expects. Mixed batches pad clips to the
longest and give text-only samples black frames. Blind configs read the
same manifests and simply ignore the media.

### Generative output heads (TTS / image OUT / video OUT)

The mirror of encoderless vision: **decoderless output**. The model's own
hidden states at `<image_out>` / `<video_out>` / `<audio_out>`
placeholders decode straight into pixels / frames / waveform — no codec,
VAE, or diffusion dependency. Each head is a **routed mixture of renderer
experts** (MoE works for these modalities exactly like for text FFNs:
renderer experts specialize — color/texture, motion, prosody — with a
switch-style load-balance loss).

```python
cfg.outgen.enabled = True
cfg.outgen.image = cfg.outgen.video = cfg.outgen.tts = True
cfg.outgen.image_token_id, cfg.outgen.video_token_id, cfg.outgen.audio_token_id = 250, 251, 252
model = RBuildModel(cfg)            # counter-verified, heads included

img = model.generate_image(ids)     # (B, 3, S, S) in [0,1]
vid = model.generate_video(ids)     # (B, F, 3, S, S) in [0,1]
wav = model.generate_audio(ids)     # (B, L) in [-1,1] @ cfg.outgen.sample_rate
```

Training uses the same manifest — add `*_out` targets and the model
learns to *produce* the media after the text:

```json
[{"text": "paint a red square", "image_out": "red.png"},
 {"text": "say hello",          "audio_out": "hello.wav"},
 {"text": "make it move",        "video_out": "move.mp4"}]
```

`manifest_batches` emits `(x, y, images, out_targets)` and `Trainer.fit`
handles it. Opt-in as always: `outgen.enabled=False` builds nothing and
the model is bit-identical to plain v3.1.

### R-OutGen — media-native model on the exact R-Build trunk

`ROutGenModel` is not a new architecture — its trunk **is** the text
model: the same `CacheLoopLine` extraction loop (critic-gated, ACT-halted,
pulling the fast-weight cache), the same `ParallelBundleStage` generative
stages (MoE branches + stage critic panels), the same `FastWeightMemory`,
`VisionTower` and `ThinkingModes`, built from the same config. Only the
text-*output* machinery is dropped (LM head and noting experts); the universal
CUA head remains, and the renderer-MoE heads are the model's **native** output:

```python
from rbuild import ROutGenModel, count_routgen_parameters, Trainer

model = ROutGenModel(cfg)         # same cfg as above — the exact text trunk
assert sum(p.numel() for p in model.parameters()) \
    == count_routgen_parameters(cfg)["total_params"]   # counter-verified

trainer = Trainer(model, cfg)     # stock Trainer — the media MSE is the objective
trainer.fit(manifest_batches("data.json", cfg))

img = model.generate_image(ids)   # hidden states -> pixels/frames/waveform
model.save_checkpoint("ckpt_routgen")          # same checkpoint format
model = ROutGenModel.load_checkpoint("ckpt_routgen")
```

Thinking modes still govern how hard the trunk thinks before it renders;
`remember()` writes facts gradient-free exactly like the text model.

### Interactive panel (Colab / Jupyter)

```python
from rbuild import interactive
ui = interactive.launch()           # widgets for every value, v3 sections included
model = ui.model
interactive.chat(model, encode, decode)
# /mode deep      switch thinking mode live      /modes     list modes
# /mode instant   reasoning off (no thinking)    /effort high   selectable effort
# /selfstats      self-learning counters         /remember  gradient-free fact write
# /nsct on        non-separate continuous training (default off)
# or tag the prompt itself:  "explain this {effort:'high'}"
```

## Stage ladder presets (continued-training path)

| preset | total (CUA included) | active/token | v3 additions |
|---|---|---|---|
| s1 | 20.6B | 3.5B | +101M critics, +19M noting |
| s2 | 93.8B | 9.4B | critics scale with stage widths |
| s3 | 122.0B | 19.4B | wider-per-token, not deeper |
| s4 | 227.0B | 40.8B | |
| s5 | 436.9B | 70.6B | |

Working-expert sizes are the v2 ladder; v3 critics, noting experts, and the
universal CUA head add their counter-verified parameters on top (`report()` splits them out).
Critics cost ~0.5% of total parameters. All presets are counter-verified;
treat them as starting points and retune in the panel.

## Training scripts

- `examples/v3_quickstart.py` — every v3 feature in one tiny CPU run:
  counter verification, ACT halting, thinking modes, self-training,
  encoderless VL + VaWU, native actuation.
- `examples/train_interactive.py` — the completely interactive trainer:
  ten numbered steps from preset to checkpoint, every input validated,
  every default recommended. Anyone can run it:
  `python examples/train_interactive.py`
- `examples/train_base_v41.py` — generic V4.1 CPT/SFT from an explicitly
  selected pretrained base; no checkpoint default.
- `examples/train_rhododendron_lite.py` — full training run for
  **rhododendron-lite (20B-A3.5B, blind)**: HF streaming data, Muon + WSD,
  bf16/fp8 flags, gradient checkpointing, periodic checkpoints, resume,
  FSDP/DDP via torchrun, optional push to the GeoThinkAI org.

See **R-Build-V4.1-DESIGN.md** for conversion/training and
**R-Build-V3-DESIGN.md** for the native architecture blueprint.

Apache-2.0. Part of the GeoThinkAI R-Build project.

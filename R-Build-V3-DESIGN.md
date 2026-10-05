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

## 3. Noting experts + non-separate continuous training (NSCT)

"Non-separate" = there is **no separate training phase and no separate
verification phase** — training and running are *simultaneous*, and the
whole thing is engineered for **low RAM** (gradient-free memory writes,
CPU fp16 note buffer). One forward pass does all of it:

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
- **Opt-in gate (v3.1):** none of this runs unless you ask for it. NSCT
  (non-separate continuous training — training and running are
  *simultaneous*, at low RAM) is controlled by `noting.nsct`;
  `False` (the default) means startup never self-trains —
  pure inference, no notes taken, nothing written. Three ways to turn it
  on: the config flag, `model.set_nsct(True)` at runtime, or
  explicitly applying a thinking mode with `self_observe=True`. The gate is
  enforced *after* the default thinking mode is applied in `__init__`, so a
  mode's `self_observe` only counts when the mode was applied by the user,
  not by default.
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

## 7. Generation watermarking

Anything the model writes can be provably traced back to it — without
changing a single weight. During sampling, the previous token is hashed
with a secret `key` to seed an RNG that splits the vocabulary into a green
list (fraction `gamma`, default 0.25); green-list logits get `+delta`
(default 2.0). Detection replays the split and runs a one-proportion
z-test on the green fraction:

- watermarked text: green fraction ≈ 0.75, z ≈ 9 (p < 1e-19)
- unwatermarked text: green fraction ≈ gamma, z ≈ 0
- wrong key: statistically indistinguishable from unwatermarked

Pure sampling-time signal — zero parameters, checkpoints untouched,
`generate(..., watermark=False)` overrides per call. Tunables live in
`WatermarkConfig` (`key`, `delta`, `gamma`, `z_threshold`); higher `delta`
watermarks harder at a small quality cost, lower `gamma` makes detection
need more tokens.

## 8. Counter verification (project invariant)

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

## 9. Backward compatibility

- `critic.enabled=False`, `noting.enabled=False`, `actuation.enabled=False`,
  `vision.enabled=False` → exact v2 architecture and parameter count.
- v2 checkpoints load: `from_dict` skips unknown/missing sections; new
  sections take defaults.
- The fast-weight memory matrix, checkpoint format, and
  `Trainer` save/load are unchanged.
- v3.1 behavior note: weights and configs are fully compatible with v3.0,
  but NSCT now defaults to OFF (`noting.nsct=False`). A v3.0 run that
  silently learned during generation stays purely inferential in v3.1
  until you opt in — set `noting.nsct=True` to restore the old behavior
  exactly. Old checkpoints/configs written with `auto_train` still load —
  the key is a legacy alias for `nsct`, and `model.set_auto_train()` is a
  legacy alias for `model.set_nsct()`.

## 10. Serving — R-Run, merged in (v3.1)

The R-Run serving engine is now part of this repo as the `rrun` package:
R-Build builds and trains the models, R-Run serves them.

- **Swap contract:** `RRunEngine.swap(model_id)` takes a swap lock, wipes
  the old KV cache and weights completely, loads the new model, allocates
  the **full** KV cache for it, and releases the lock. The HTTP server
  never restarts — the port stays bound and clients keep their
  connections; only the resident model changes.
- **KV discipline:** `KVCacheManager` sizes the cache from the model's real
  geometry (`2 × n_layers × n_kv_heads × head_dim × dtype_bytes` per token)
  and always rebuilds it at full capacity on every swap — never shrunk for
  speed.
- **Backends:** `rbuild` (native `Trainer.save_checkpoint` directories via
  `Trainer.load_checkpoint` — byte-level UTF-8 tokenization by default,
  optional HF tokenizer, watermark passthrough, KV sized from R-Build
  geometry: loop blocks + every branch of every stage), `vllm` (GPU fast
  path), `hf` (transformers, CPU-friendly), `mock` (no weights, smoke
  tests).
- **API:** OpenAI-compatible `/v1/chat/completions` + `/v1/models`, admin
  `/admin/swap` / `/admin/wipe` / `/admin/status`. CLI:
  `rrun serve <model> --backend rbuild`, `rrun swap <model>`,
  `rrun status`, `rrun wipe`.
- Serving is pure inference by default — the NSCT gate (§3) applies
  to served models too, so a resident model never self-trains unless the
  operator opts in.

## 11. Data — manual JSON manifest + no-fuss HF image/video (`rbuild/data.py`)

One data layer for every model type. Generators yield `(x, y)` for blind
configs and `(x, y, images)` for vision configs; `Trainer.fit()` accepts
both batch shapes.

- **Manual JSON manifest (any model type):** a `.json` list or `.jsonl`
  file of `{"text": ..., "images": [...], "video": ...}` entries. Media
  values are local paths or https URLs; images may also be `{"path"|"bytes"}`
  dicts. Blind configs load the same manifest and ignore media (one-time
  notice) — the manifest format is universal.
- **Placeholder splicing:** each sample's token stream is one run of
  `vision.image_token_id` placeholders of exactly the length the vision
  tower will emit (`n_frames × tokens_per_image`, plus VaWU whole-video
  tokens when enabled), followed by byte-level UTF-8 text. Truncation
  never cuts the placeholder run — if the run alone exceeds `seq`, a
  clear error says to raise `seq`.
- **Mixed batches:** clips pad to the batch's longest frame count by
  repeating the last frame; text-only samples in a vision batch get black
  (zero) frames so every sample has the same placeholder count K that
  `_splice_vision` requires.
- **Video decode fallback chain:** decord → imageio → ffmpeg/ffprobe
  subprocess. None is mandated; whatever is installed is used. Frames are
  sampled uniformly over the clip and resized to `vision.image_size`.
- **HF streaming (no-fuss):** `hf_image_batches` / `hf_video_batches`
  stream any dataset (`load_dataset(..., streaming=True)`), auto-detect
  the text/image/video columns from the first row (PIL objects, media
  paths/URLs, or `{"bytes"|"path"}` dicts), and print the mapping once.
  Override with `text_column=` / `image_column=` / `video_column=`.
  Rows whose media fails to decode are skipped with a one-time notice.

## 12. Generative output heads — TTS, image OUT, video OUT (`rbuild/outgen.py`)

The output mirror of v3's encoderless vision: **decoderless output**. The
LLM's own hidden states do the rendering — a head reads the hidden states
at its placeholder run (`<image_out>` / `<video_out>` / `<audio_out>`
token ids, appended after the prompt) and decodes them straight into
pixels, frames, or waveform. No external codec, VAE, or diffusion stack;
one transformer, media in and media out.

**Is MoE usable for these modalities?** Yes — the same way it is for text
FFNs, and this module is the proof. Each head decodes through a **routed
mixture of renderer experts** (`RendererMoE`): a learned router sends
every output token to its top-k renderer experts (2-layer MLPs to the
modality's output dim), with a switch-style load-balance aux loss.
Renderer experts specialize the way working experts do — color/texture
experts in the image head, motion experts in the video head, prosody
experts in the TTS head. (The same pattern appears in production
T2I/TTS/video backbones: MoE feedforwards in AR TTS transformers, MoE
DiT blocks in video generators, mixture-of-denoising-experts in T2I.)

- **ImageOutHead:** `grid²` placeholders → per-token renderer → patch
  pixels → unpatchified to `(B, 3, S, S)`, learned 2-D position table.
- **VideoOutHead:** `F × grid²` placeholders → shared renderer, with a
  frame-position table plus a patch-position table → `(B, F, 3, S, S)`.
- **TTSOutHead:** `audio_tokens` placeholders → renderer → waveform
  chunks of `audio_chunk` samples, `tanh`-bounded → `(B, L)` at
  `outgen.sample_rate`.
- **Training:** `forward(..., out_targets={"image"|"video"|"audio": T})`
  adds `loss_weight`-scaled MSE against the decoded output (normalized
  pixels `[0,1]`, waveform `[-1,1]`), per head, only over samples whose
  input carries that head's placeholder run — mixed manifests are safe.
  Out-placeholder positions never enter the text CE (masked like vision
  positions). The balance loss rides along at `moe_balance` weight.
- **Generation:** `model.generate_image/video/audio(prompt_ids)` appends
  the head's placeholder run, encodes once, decodes. No autoregression
  over pixels — one forward paints the whole canvas (thinking modes and
  ACT halting still govern how hard the transformer thinks first).
- **Data:** manifest entries add `"image_out"`, `"audio_out"`,
  `"video_out"` (paths or URLs; `.wav` via the stdlib, anything else via
  ffmpeg). `manifest_batches` emits `(x, y, images, out_targets)`;
  `Trainer.fit` accepts 2/3/4-tuples unchanged otherwise.
- **Counter-verified:** router + R renderer experts + position tables are
  mirrored exactly in `count_parameters()` (`outgen_params` in the
  report); `enabled=False` builds nothing — bit-identical v3.1.

## 13. R-OutGen — the media-native model on the exact text trunk (`rbuild/routgen.py`)

`ROutGenModel` answers "what does a pure R-Build generator look like"
with: **the same architecture**. Its trunk is not a re-implementation —
it imports and builds the very classes the text model uses:

```
prompt tokens (text, optional vision soft tokens)
  -> embed
  -> CacheLoopLine        (the extraction loop: critic-gated, ACT-halted,
                           pulling the fast-weight cache — same class)
  -> ParallelBundleStage x N  (fine-grained MoE branches + stage critic
                               panels — same class)
  -> final_norm
  -> OutGen heads         (renderer MoE — the NATIVE output)
```

- **Exact trunk, same config.** One `RBuildConfig` drives both
  `RBuildModel` and `ROutGenModel`; validation, init scheme, thinking
  modes (`model.thinking_mode.deep()` governs how hard it thinks before
  rendering), ACT halting, and `remember()` gradient-free fact writes all
  behave identically.
- **Only text-output machinery is dropped:** the LM head, the actuation
  action head, and the noting experts — all text-vocabulary devices. The
  media MSE + renderer balance loss is the sole training objective; there
  is deliberately no text CE (`forward` accepts `targets=` for
  `Trainer.fit` compatibility and ignores it, raising if it is the only
  supervision given).
- **Counter derivation (project invariant kept):**
  `count_routgen_parameters(cfg)` starts from the canonical
  `cfg.count_parameters()` and subtracts exactly the dropped pieces
  (noting + actuation + untied LM head when `tie_embeddings=False`), so
  the printed number matches `sum(p.numel())` on a built ROutGenModel —
  verified in the smoke suite alongside a same-config `RBuildModel` to
  prove the delta is precisely the text-output params.
- **Stock Trainer, stock checkpoints.** `Trainer(model, cfg).fit(...)`
  trains it unchanged from `*_out` manifest batches; checkpoints keep the
  same `model.pt + rbuild_config.json + meta.json` format (with
  `model_class: ROutGenModel` in meta), reloadable via
  `ROutGenModel.load_checkpoint`.
- **Relationship to §12:** `RBuildModel + outgen` is one model that does
  text *and* media; `ROutGenModel` is the media-only build — no text
  head, no text loss, the renderer MoE as the entire output side.

# R-Build v4.1 — Architecture Conversion and Continued-Training Blueprint

**Status:** implemented additively on the `v4` branch.
**Base policy:** every base checkpoint remains selectable and modifiable.
**Compatibility:** the complete native V3/V4 engine remains available; no feature
or preset is removed.

## 0. Thesis

V4.1 replaces mandatory from-scratch construction with a pretrained continuation
path. Any PyTorch/Hugging Face causal-LM architecture can become an R-Build
system through a non-destructive adapter, then improve through continued
pretraining (CPT) and supervised fine-tuning (SFT).

The conversion is deliberately not destructive model surgery. Automatically
replacing an arbitrary model's attention, MoE, normalization, or output layers
would invalidate its checkpoint and could remove capabilities. V4.1 therefore
uses two supported paths:

| path | trunk | R-Build capabilities |
|---|---|---|
| **Native** | original R-Build MoD → fine-grained MoE → shared experts → critic workflow | complete V3/V4 engine |
| **Adapted** | the selected pretrained architecture, unchanged | additive critic panel, fast-weight memory, common CPT/SFT, checkpoints, release gates |

The original expert workflow is preserved in the native engine. An adapted base
keeps its own native computation exactly; R-Build modules never silently rewrite
its logits or generation path.

## 1. Base-model contract

`RBuildAdaptedModel(base_model, V41Config)` accepts an already constructed
PyTorch model. `RBuildAdaptedModel.from_pretrained(base_model_id, ...)` loads a
caller-selected Hugging Face causal LM.

There is no global default checkpoint. Qwen3-235B-A22B appears only as the
default `--base-model` in `examples/train_rhododendron_v41.py`, because that is
Rhododendron's selected starting point. Passing another `--base-model` changes
the base without changing R-Build.

Construction must not modify base parameters. `forward()` and `generate()` are
exact delegations to the base, so merely wrapping a model cannot degrade its
output.

## 2. Additive R-Build capabilities

### Critic panel

The adapter can construct X critic experts over the base hidden width. The panel
uses the original `CriticPanel` implementation and Y-vote rule. It gates explicit
fast-weight writes; it does not intercept base logits.

### Fast-weight memory

The adapter uses the original delta-rule key→value matrix. `remember_hidden()`
detaches hidden states, optionally requires Y critic votes, and writes approved
key/value pairs without gradients. `recall_hidden()` reads the matrix. Memory and
critic states persist alongside the base checkpoint.

### Existing native capabilities

The V4 discrete-diffusion draft head, exact-AR verification, ACT extraction,
noting experts/NSCT, thinking modes, encoderless vision/VaWU, output media heads,
native actuation, watermarking, parameter counters, and R-Run serving remain in
their existing modules. V4.1 deletes none of them.

## 3. Continued pretraining

`continued_pretraining_batches()` tokenizes raw documents, optionally inserts
EOS, packs fixed-length causal-LM windows, and yields:

```text
input_ids       = packed token window
labels          = identical token window
attention_mask  = 1 for every packed token
```

The base model performs its normal internal next-token shift. CPT updates the
selected pretrained weights rather than initializing a replacement from scratch.

## 4. Supervised fine-tuning

`supervised_fine_tuning_batches()` accepts `(prompt, response)` pairs or mappings
with those keys. It truncates the prompt first when space is limited, appends EOS
to the response when available, and emits:

```text
input_ids      = prompt + response + padding
labels         = -100 for prompt/padding, token ids for response
attention_mask = 1 for real tokens, 0 for padding
```

This prevents the instruction tokens from becoming the supervised target. Full
base training is the default; `freeze_base()` enables adapter-only fine-tuning.

## 5. Original training-speed workflow

`RBuildBaseTrainer` reuses the existing optimizer split and WSD scheduler:

- Muon for eligible 2-D hidden matrices;
- AdamW for embeddings, output heads, routers/gates, norms, and 1-D parameters;
- warmup-stable-decay scheduling;
- gradient accumulation and clipping;
- fp32/bf16 and the existing fp8 portability contract;
- periodic base + adapter + optimizer checkpoints and resume state.

CPT and SFT can run sequentially from the same selected base. Checkpoints contain
the base weights, R-Build configuration, critic/memory state, and trainer step.

## 6. No-capability-loss release contract

“No capability loss” is a tested release requirement, not an assumption about
training. Two separate checks apply:

1. **Conversion invariant:** before training, adapted disabled-path outputs must
   exactly equal base outputs and construction must leave base weights unchanged.
2. **Post-training canaries:** every release must run base-capability evaluations
   appropriate to that checkpoint. `v41_release_gate()` rejects exact-output
   mismatches and canary regressions beyond the selected tolerance.

A CPT/SFT loss decrease alone is insufficient evidence that general, reasoning,
multilingual, tool, vision, or other base capabilities were preserved.

## 7. Acceptance tests

`tests/v41_test.py` covers:

- exact forward/generation passthrough and no base surgery;
- critic-gated memory and checkpoint recovery;
- CPT batch construction;
- SFT prompt masking;
- real CPT and SFT optimizer steps;
- capability-regression gate failures.

The existing V3/V4 suites remain required, proving that V4.1 did not remove or
break the native engine.

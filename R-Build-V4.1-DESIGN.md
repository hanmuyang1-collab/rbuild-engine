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
| **Native** | original R-Build MoD → fine-grained MoE → shared experts → critic workflow | complete V3/V4 engine and universal CUA |
| **Adapted** | the selected pretrained architecture, unchanged | additive critic panel, fast-weight memory, universal CUA, common CPT/SFT, checkpoints, release gates |

The original expert workflow is preserved in the native engine. An adapted base
keeps its own native computation exactly; R-Build modules never silently rewrite
its logits or generation path.

## 1. Base-model contract

`RBuildAdaptedModel(base_model, V41Config)` accepts an already constructed
PyTorch model. `RBuildAdaptedModel.from_pretrained(base_model_id, ...)` loads a
caller-selected Hugging Face causal LM.

There is no global or project-specific default checkpoint. The generic
`examples/train_base_v41.py` entry point requires `--base-model`; selecting a
different checkpoint does not change R-Build.

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

### Universal CUA

`RBuildModel`, `ROutGenModel`, and `RBuildAdaptedModel` always construct the same
`ActionCodec`, dedicated `ActuationHead`, and `ComputerUseAgent` runtime as
`model.cua`. The common contract provides:

- `observe(...)` / `propose(hidden_states)` for native hidden-state action proposals;
- `decode(token_ids)` for reserved click, scroll, type, and wait tokens;
- `action_loss(hidden_states, action_token_ids)` for supervised CUA trajectories;
- `execute(actions, driver)` for real caller-provided OS drivers;
- `act(hidden_states, driver)` for a complete propose-and-execute step.

Native text models merge action columns into ordinary generation logits by default.
`actuation.enabled=False` retains legacy text-logit dimensions but leaves the
separate CUA head/runtime intact. R-OutGen keeps its renderer outputs and CUA head.
Adapted models keep exact base `forward()`/`generate()` delegation and persist the
CUA head in the adapter checkpoint.

An initialized action head is not evidence of computer-use competence: the model
must be trained and evaluated on grounded CUA trajectories. Real operations occur
only through an explicit runtime driver. If a batch is marked irreversible, any
action has `irreversible=True`, uncertainty uses `on_uncertain="confirm"`, or an
application policy requires confirmation, execution requires `confirmed=True`.
All policy and confirmation guards run before the first side effect and are
independent of thinking effort.

#### Uncertainty and critic verification

`propose()` computes action confidence and entropy from the existing action logits.
Thresholds are evaluated independently for every batch row. Uncertain rows either
receive the existing `wait` token (`on_uncertain="wait"`) or retain their proposal
with a confirmation requirement (`on_uncertain="confirm"`).

`propose_verified()` takes top-k candidates from the same action head, conditions
candidate hidden states with existing action-head weight rows, and scores them with
the model's existing native or adapted critic panel. Approval combines Y-votes and
satisfaction per row; if no candidate passes, that row emits `wait`. The returned
`CUADecision.verification` contains candidate tokens, log probabilities, votes,
satisfaction, approval masks, and the selected candidate.

`act_verified()` passes `expected_postcondition` to execution. A driver must implement
`verify(expectation)`; false or unsupported verification raises
`PostconditionFailed`. Postconditions run after actions and do not roll back partial
side effects, so they complement rather than replace pre-execution policy guards.

#### Event observation, shadow execution, and policies

`ScreenChangeDetector` creates a small downsampled screen signature (or a digest for
byte observations). `observe_if_changed()` avoids model invocation until the screen
changes beyond the configured threshold, subject to minimum and maximum polling
intervals. This is parameter-free and does not alter model outputs.

`ShadowDriver` records `moveTo`, `click`, `scroll`, `write`, and verification calls.
It never invokes methods on its optional wrapped real driver. `act(..., shadow=True)`
and `act_verified(..., shadow=True)` therefore support dry runs without OS effects.

`CUAPolicy` supports action-kind allowlists, x/y bounds, scroll and text-length
limits, blocked text, per-kind confirmation, and minimum execution intervals.
`PolicyDriver` attaches the policy to a driver. `execute()` validates the whole batch
before its first action and advances rate-limit state only after successful
execution. Policy rejection and missing confirmation raise before side effects.

#### Deterministic auditing and verified trajectory replay

Every `execute()` outcome can be written to `CUAReplayLedger`: executed, rejected,
or failed, together with actions, confirmation state, driver type, postcondition,
metadata, and error. The in-memory ledger is bounded; optional JSONL persistence
rotates at a byte limit. `load_jsonl()` restores records, and `replay()` executes
the exact recorded actions through the normal policy/confirmation path rather than
sampling fresh model outputs.

`CUATrajectory` pairs `(1,T,D)` hidden states with `(1,T)` reserved action labels.
`CUATrajectoryBuffer` rejects unverified examples, stores hidden states on CPU in
fp16 and labels as int64, enforces trajectory-count and byte budgets, and evicts
oldest examples. `batches()` restores hidden states to fp32 and pads labels with
`-100`, which `action_loss()` ignores. Replay is a training-data mechanism, not a
claim that stored actions remain valid on a changed live screen.

`build_recovery_curriculum()` derives variants only from verified trajectories. It
covers `unexpected_dialog`, `scrolled_target`, `loading_delay`, `stale_screenshot`,
and `failed_click`, using existing wait/click tokens and hidden states; it adds no
model weights.

#### Frozen-base training, screenshots, and application adapters

`freeze_for_cua()` freezes every existing model parameter, then unfreezes only the
action head and any opt-in screenshot projector/application adapters.
`RBuildBaseTrainer.fit_cua()` consumes verified replay batches and reuses the
existing autocast, gradient accumulation/clipping, Muon/AdamW split, WSD schedule,
checkpointing, and logging. The optimizer now handles Muon-only frozen-base runs
without constructing an empty AdamW parameter list.

Adapted models may set `cua_screenshot_enabled=True`.
`EncoderlessScreenshotProjector` is one bias-free `Conv2d` patch projection from
pixels to base hidden width—no second encoder, positional table, or transformer.
`screenshot_hidden()` accepts `(B,C,H,W)` or `(B,N,C,H,W)` and returns patch
hidden states. `observe_screenshot()` proposes from the final patch position by
default; callers can override `position`. This path is CUA-only and never rewrites
adapted base `forward()` or `generate()`.

`add_cua_adapter(name, rank, alpha)` adds a named low-rank residual over existing
CUA hidden states. Its up projection is zero-initialized, so a new adapter initially
preserves decisions exactly. `activate_cua_adapter()` selects or disables it.
Adapter weights, rank/alpha specifications, active name, screenshot projector, and
action head persist in adapter checkpoints. Older checkpoints load through missing
field defaults. Native and R-OutGen parameter counters remain unchanged because all
runtime/replay additions are parameter-free and the new trainable modules are
adapted-model-only and opt-in.

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
training. Three separate checks apply:

1. **Conversion invariant:** before training, adapted disabled-path outputs must
   exactly equal base outputs and construction must leave base weights unchanged.
2. **Post-training canaries:** every release must run base-capability evaluations
   appropriate to that checkpoint. `v41_release_gate()` rejects exact-output
   mismatches and canary regressions beyond the selected tolerance.
3. **CUA contract and grounded canaries:** `run_cua_contract_canaries()` checks
   action loss, fake-driver click/scroll/type execution, interruption curriculum,
   wrong-screen detection, policy rejection, postconditions, and pre-side-effect
   irreversible-action rejection across supplied model classes. `cua_release_gate()`
   requires all selected canaries and safety-critical checks. Real application
   releases must additionally supply grounded task canaries; the built-in contract
   checks do not establish operator competence.

A CPT/SFT/CUA loss decrease alone is insufficient evidence that general, reasoning,
multilingual, tool, vision, computer-use, or other base capabilities were preserved.

## 7. Acceptance tests

`tests/v41_test.py` covers:

- exact forward/generation passthrough and no base surgery;
- critic-gated memory and checkpoint recovery;
- CPT batch construction;
- SFT prompt masking;
- real CPT and SFT optimizer steps;
- capability-regression gate failures.

`tests/cua_test.py` covers all three model classes, native proposals and action
loss, legacy text-logit compatibility, pre-universal-CUA checkpoint recovery,
fake-driver side effects, and pre-side-effect irreversible-action confirmation.

`tests/cua_expansion_test.py` adds 11 acceptance tests covering all 12 expansion
items: uncertainty abstention/confirmation, per-row critic verification and
postconditions, event observation, shadow mode, application policies/rate limits,
JSONL auditing and deterministic replay, verified-only bounded trajectory replay,
all recovery events, cross-model release gates, frozen-base training with
byte-identical base weights, encoderless screenshot input with exact base logits,
application-adapter checkpoint recovery, and unchanged native/R-OutGen parameter
counters. Tests use fake drivers only.

The existing V3/V4 and complete smoke suites remain required, proving that V4.1
did not remove or break the native engine.

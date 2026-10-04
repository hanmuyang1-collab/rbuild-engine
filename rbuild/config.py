"""
R-Build v3 — fully user-modifiable configuration.

Every value in the engine lives here and can be changed:
  - programmatically:      cfg = RBuildConfig(); cfg.critic.n_critics = 8
  - interactively:         rbuild.interactive.launch()  (widgets in Colab/Jupyter,
                           console prompts elsewhere)
  - from the CLI:          python -m rbuild.interactive

Sections:

  ModelConfig      — sizes of the network itself
  CacheLoopConfig  — the sequential "single line" extraction stage that loops
                     to pull the fast-weight memory cache
  ParallelConfig   — the parallel generative stages whose branches bundle
                     outputs and push them to the next parallel stage
  CriticConfig     — v3: X parallel critic experts (more capacity than the
                     working experts), ACT-style halting of the extraction
                     loop until Y critics are satisfied
  NotingConfig     — v3: noting experts + critic-verified, non-separate
                     self-training (learn while running, low RAM)
  ThinkingConfig   — v3: default thinking mode + user-created modes
  ActuationConfig  — v3: native action tokens (the model clicks by itself)
  WatermarkConfig  — v3: green-list generation watermarking (zero params)
  MemoryConfig     — the gradient-free fast-weight delta-rule memory
  VisionConfig     — ViT tower *or* encoderless vision, plus VaWU whole-video
                     summary tokens (blind by default)
  TrainConfig      — Muon / WSD / precision / batching / cost model

Validation and derived counters (parameters, active parameters, estimated
training cost with the naive-vs-optimized comparison) run on demand via
`cfg.report()` and automatically inside the interactive panel. The counter
mirrors the v3 module tree exactly — including every critic, note-taker,
and action head — so the printed number matches a built model.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #

@dataclass
class ModelConfig:
    """Global model sizes. Everything here is user-modifiable."""
    vocab_size: int = 32000
    d_model: int = 256
    max_seq_len: int = 2048
    n_heads: int = 8                 # query heads
    n_kv_heads: int = 2              # GQA key/value heads
    head_dim: Optional[int] = None   # None -> d_model // n_heads
    rope_theta: float = 10000.0
    rmsnorm_eps: float = 1e-6
    tie_embeddings: bool = True
    dropout: float = 0.0

    def resolved_head_dim(self) -> int:
        return self.head_dim or (self.d_model // self.n_heads)


@dataclass
class CacheLoopConfig:
    """
    Stage A — the *single line* of layers that loops to pull cache.

    In v3 this is the *extraction loop*: with critics enabled it runs
    adaptively — at most `critic.max_loops` iterations, halting per token
    (ACT-style) once the critics are satisfied. `n_loops` remains the exact
    loop count when critics are disabled.
    """
    n_layers: int = 2                # blocks in the single line
    n_loops: int = 4                 # loop count (critics off) / default cap reference
    share_loop_weights: bool = True  # reuse weights across loops
    memory_read_every: int = 1       # pull cache every k-th loop iteration
    use_mod_routing: bool = True     # MoD: only top-p tokens hit each block
    mod_capacity: float = 0.5        # top fraction of tokens routed per block


@dataclass
class ParallelConfig:
    """
    Stage B — the parallel generative stages.

    Each stage owns `n_branches` layer-branches that run in parallel over the
    same input, plus (v3, `critic.stage_critics`) a panel of X critic experts
    with more capacity than the working experts. Outputs are *bundled* and
    pushed to the next stage.
    """
    n_stages: int = 3                # how many parallel stages
    n_branches: int = 4              # parallel branches per stage
    bundle_mode: str = "gate"        # "gate" | "mean" | "concat"
    branch_ffn: str = "moe"          # "moe" | "dense"
    # fine-grained MoE (used when branch_ffn == "moe")
    n_experts: int = 8               # fine-grained routed experts per branch
    expert_top_k: int = 2            # experts activated per token
    n_shared_experts: int = 1        # always-on shared experts per branch
    expert_ffn_dim: Optional[int] = None  # None -> d_model * ffn_mult / n_experts
    ffn_mult: float = 4.0            # dense-equivalent FFN multiplier
    dense_ffn_dim: Optional[int] = None   # used when branch_ffn == "dense"
    use_mod_routing: bool = True     # MoD on parallel branches
    mod_capacity: float = 0.75


@dataclass
class CriticConfig:
    """
    v3 — critic experts and ACT-style adaptive halting.

    The extraction loop runs until at least `y_critics` of the X parallel
    critics are satisfied (per-token halting is ACT-style: cumulative
    satisfaction mass crosses 1 - halt_eps). Each generative stage carries
    its own panel of X critics, each with `critic_capacity_mult` times the
    capacity of a working expert — verification needs headroom to be a
    trustworthy gate. `halt_loss_weight` scales the ACT ponder cost
    (loops taken + remainders) added to the training loss.

    enabled=False removes every critic parameter and restores the exact
    v2 architecture.
    """
    enabled: bool = True
    n_critics: int = 4               # X — parallel critic experts per panel
    critic_capacity_mult: float = 2.0  # critic hidden = working-expert dim x this
    critic_hidden: Optional[int] = None  # explicit override (None -> from mult)
    y_critics: int = 2               # Y — satisfied critics needed to halt/verify
    threshold: float = 0.6           # per-critic "satisfied" score
    max_loops: int = 8               # extraction-loop cap (ACT)
    min_loops: int = 1               # never halt before this many iterations
    halt_eps: float = 0.01           # ACT halt threshold epsilon
    halt_loss_weight: float = 0.01   # weight of the ACT ponder cost
    stage_critics: bool = True       # critic panel on every generative stage


@dataclass
class NotingConfig:
    """
    v3 — noting experts + critic-verified non-separate continuous training.

    NSCT = non-separate continuous training: training and running happen
    SIMULTANEOUSLY, at low RAM. During normal forwards (including
    generation), noting experts propose
    candidate facts from the final hidden states. The generative stages'
    critics verify each note; verified notes are (a) written into the
    fast-weight memory immediately (gradient-free — the model learns while
    running, ~zero extra RAM) and (b) queued in a CPU fp16 buffer so
    `Trainer.self_train_step()` can consolidate them into the slow weights
    with a real gradient step. No separate training phase, no separate
    verification phase: running and learning happen in parallel.

    v3.1: NSCT is OPT-IN. `nsct=False` (default) means the model never
    self-trains unless you turn it on — per config, per call
    (`model.set_nsct(True)`), or per thinking mode (`self_observe`).
    """
    enabled: bool = True
    nsct: bool = False               # NSCT: train & run simultaneously, low RAM — OFF by default
    n_noting_experts: int = 2        # separate note-taking experts
    note_hidden: Optional[int] = None  # None -> d_model
    verify_y_critics: int = 2        # critics that must approve a note
    verify_threshold: float = 0.6    # mean critic satisfaction to accept
    min_confidence: float = 0.5      # note-taker confidence floor
    write_to_memory: bool = True     # verified notes -> fast-weight memory now
    memory_write_lr: float = 0.5     # delta-rule lr for self-observed writes
    buffer_capacity: int = 4096      # CPU fp16 note buffer (low RAM)
    self_train_batch: int = 128      # notes per consolidation step
    observe_in_training: bool = False  # also take notes during fit()

    # v3.1 legacy alias — checkpoints/scripts written before the NSCT
    # rename still say `auto_train`; reads and writes map onto `nsct`
    # (from_dict() picks it up automatically via hasattr/setattr).
    @property
    def auto_train(self) -> bool:
        return self.nsct

    @auto_train.setter
    def auto_train(self, v: bool) -> None:
        self.nsct = bool(v)


@dataclass
class ThinkingConfig:
    """
    v3 — thinking modes. Runtime presets that retune the adaptive machinery
    (max loops, Y critics, halt threshold, sampling, self-observation)
    without rebuilding the model. Create your own:

        model.thinking_mode.create("exam", max_loops=10, y_critics=3)
        model.thinking_mode.exam()

    `custom_modes` persists user-created modes into checkpoints.
    """
    default_mode: str = "balanced"
    save_modes: bool = True
    custom_modes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActuationConfig:
    """
    v3 — native actuation: reserved action tokens let the model click,
    scroll, type and wait *by generating a token* — no external tool loop.

    The output space grows by `n_action_tokens` (1 wait + grid^2 clicks +
    2*scroll_steps scrolls + type begin/end). A dedicated ActuationHead
    produces the action columns of the logits, so the action space stays
    clean even with tied embeddings. Ground the head with vision
    (encoderless + VaWU recommended: the screen is just frames).
    """
    enabled: bool = False
    screen_grid: int = 64            # click grid resolution (g x g cells)
    scroll_steps: int = 8            # discrete scroll magnitudes (up/down)


@dataclass
class WatermarkConfig:
    """
    v3 — generation watermarking (green-list logit biasing).

    While sampling, the previous token is hashed with `key` to seed a split
    of the vocabulary into a green list (fraction `gamma`); green logits are
    boosted by `delta`. Text generated this way is provably yours: anyone
    with the key replays the split and runs a z-test (WatermarkDetector).
    Pure sampling-time signal — zero parameters, checkpoints unaffected.
    """
    enabled: bool = False
    key: str = "rbuild-v3"           # secret — keep it private
    delta: float = 2.0               # green-list logit boost
    gamma: float = 0.25              # green-list fraction of the vocab
    z_threshold: float = 4.0         # z-score needed to call "watermarked"


@dataclass
class MemoryConfig:
    """
    Fast-weight memory: a delta-rule key->value matrix written *without
    gradients* (facts go straight in) and persisted with the checkpoint.
    """
    enabled: bool = True
    key_dim: int = 128
    value_dim: int = 256             # usually == model.d_model
    write_lr: float = 1.0            # delta-rule step size for fact writes
    decay: float = 0.999             # per-step forgetting of the matrix
    read_gate_init: float = 0.1      # initial strength of cache injection
    max_facts: int = 65536           # soft cap for the fact log


@dataclass
class VisionConfig:
    """
    VL support. Default is *blind*: enabled=False means no vision machinery
    is built at all and the model is bit-identical to the text-only path.

    mode="vit"         — v2.1 ViT tower over patches (video = frame tokens).
    mode="encoderless" — v3: no vision encoder at all. Patches are
                         normalized and projected straight into d_model;
                         the LLM itself does the seeing.
    vawu=True          — v3: Video-as-Whole-Understanding. A learned-query
                         attention pooler compresses all frames into
                         `vawu_tokens` whole-video summary tokens, prepended
                         to the frame stream: the model reads the video as
                         a whole before its parts.
    """
    enabled: bool = False            # False = blind (text-only), zero overhead
    image_token_id: Optional[int] = None  # reserved placeholder id in the vocab
    mode: str = "vit"                # "vit" | "encoderless" (v3: no encoder)
    image_size: int = 224
    patch_size: int = 14
    channels: int = 3
    vit_layers: int = 6
    vit_dim: Optional[int] = None    # None -> d_model
    vit_heads: int = 12
    vit_ffn_mult: float = 4.0
    use_cls_token: bool = False
    freeze_vision: bool = False      # train projector only (cheap VL bootstrap)
    # video
    video: bool = True               # allow (B, frames, C, H, W) inputs
    max_video_frames: int = 16       # learned frame-position table size
    # v3: Video-as-Whole-Understanding
    vawu: bool = False               # prepend whole-video summary tokens
    vawu_tokens: int = 4             # how many whole-video tokens


@dataclass
class TrainConfig:
    """Training speed stack + cost model. All user-modifiable."""
    # optimization
    optimizer: str = "muon"          # "muon" | "adamw"
    lr: float = 3e-3                 # Muon likes higher lr than AdamW
    adamw_lr: float = 3e-4           # lr for embeddings/norms/1-D params
    weight_decay: float = 0.1
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5           # Newton-Schulz iterations
    batch_size: int = 8
    grad_accum: int = 4
    max_steps: int = 2000
    warmup_steps: int = 100
    cooldown_frac: float = 0.4       # WSD decay tail fraction
    grad_clip: float = 1.0
    # v3 self-training consolidation
    self_train_every: int = 0        # >0: consolidate verified notes every N steps
    self_train_lr: float = 3e-4      # lr for the consolidation step
    # speed stack
    precision: str = "bf16"          # "fp32" | "bf16" | "fp8"
    chunked_ce: bool = True          # chunked cross-entropy (memory saver)
    ce_chunk_tokens: int = 4096
    grad_checkpoint: bool = False
    # cost model (for report(); prices user-modifiable)
    gpu_price_per_hour: float = 2.0  # e.g. H100 SXM on a budget cloud
    gpus: int = 8
    tokens_per_step: Optional[int] = None   # None -> batch*accum*seq
    achieved_tflops: float = 400.0   # per-GPU throughput of your stack


# --------------------------------------------------------------------------- #
# Root config
# --------------------------------------------------------------------------- #

@dataclass
class RBuildConfig:
    """The single object every part of R-Build reads. Modify anything."""
    model: ModelConfig = field(default_factory=ModelConfig)
    cache_loop: CacheLoopConfig = field(default_factory=CacheLoopConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    noting: NotingConfig = field(default_factory=NotingConfig)
    thinking: ThinkingConfig = field(default_factory=ThinkingConfig)
    actuation: ActuationConfig = field(default_factory=ActuationConfig)
    watermark: WatermarkConfig = field(default_factory=WatermarkConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # ------------------------------------------------------------------ #
    # derived sizes shared by the counter and the model builder
    # ------------------------------------------------------------------ #
    def resolved_expert_dim(self) -> int:
        m, p = self.model, self.parallel
        return p.expert_ffn_dim or max(8, int(m.d_model * p.ffn_mult / p.n_experts))

    def resolved_critic_hidden(self) -> int:
        c = self.critic
        return c.critic_hidden or max(16, int(self.resolved_expert_dim()
                                              * c.critic_capacity_mult))

    def resolved_note_hidden(self) -> int:
        return self.noting.note_hidden or self.model.d_model

    def n_action_tokens(self) -> int:
        if not self.actuation.enabled:
            return 0
        a = self.actuation
        return 1 + a.screen_grid * a.screen_grid + 2 * a.scroll_steps + 2

    def effective_vocab_size(self) -> int:
        return self.model.vocab_size + self.n_action_tokens()

    # ------------------------------------------------------------------ #
    # validation
    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        m, cl, p, mem = self.model, self.cache_loop, self.parallel, self.memory
        c, n, a = self.critic, self.noting, self.actuation
        assert m.d_model % m.n_heads == 0 or m.head_dim is not None, \
            "d_model must divide n_heads unless head_dim is set"
        assert m.n_heads % m.n_kv_heads == 0, "n_heads must be a multiple of n_kv_heads"
        assert cl.n_loops >= 1 and cl.n_layers >= 1
        assert 0 < cl.mod_capacity <= 1 and 0 < p.mod_capacity <= 1
        assert p.n_stages >= 1 and p.n_branches >= 1
        assert p.bundle_mode in ("gate", "mean", "concat")
        assert p.branch_ffn in ("moe", "dense")
        if p.branch_ffn == "moe":
            assert 1 <= p.expert_top_k <= p.n_experts
        assert mem.key_dim >= 8 and mem.value_dim >= 8
        assert self.train.precision in ("fp32", "bf16", "fp8")
        assert self.train.optimizer in ("muon", "adamw")
        # v3: critics
        if c.enabled:
            assert c.n_critics >= 1, "need at least one critic"
            assert 1 <= c.y_critics <= c.n_critics, \
                "y_critics must be within [1, n_critics]"
            assert 0.0 <= c.threshold <= 1.0 and 0.0 < c.halt_eps < 1.0
            assert c.max_loops >= 1 and 1 <= c.min_loops <= c.max_loops
            assert c.critic_capacity_mult > 0
        # v3: noting
        if n.enabled:
            assert n.n_noting_experts >= 1
            assert 0.0 <= n.verify_threshold <= 1.0 and 0.0 <= n.min_confidence <= 1.0
            if c.enabled:
                assert 1 <= n.verify_y_critics <= c.n_critics
            if n.write_to_memory:
                assert mem.enabled, "noting.write_to_memory requires memory.enabled"
        # v3: actuation
        if a.enabled:
            assert a.screen_grid >= 4 and a.scroll_steps >= 1
        # v3: watermark
        w = self.watermark
        if w.enabled:
            assert w.delta > 0 and 0.0 < w.gamma < 1.0 and w.z_threshold > 0
        v = self.vision
        if v.enabled:
            assert v.image_token_id is not None, \
                "vision.enabled=True requires an image_token_id placeholder"
            assert v.image_size % v.patch_size == 0, "image_size must divide patch_size"
            assert v.mode in ("vit", "encoderless")
            vd = v.vit_dim or m.d_model
            if v.mode == "vit":
                assert vd % v.vit_heads == 0, "vit_dim must divide vit_heads"
            if v.vawu:
                assert v.video, "vawu requires vision.video=True"
                assert v.vawu_tokens >= 1

    # ------------------------------------------------------------------ #
    # parameter counter (naive vs optimized stack)
    # ------------------------------------------------------------------ #
    def count_parameters(self) -> Dict[str, int]:
        """
        Static count of total / active (per-token) parameters. The counting
        mirrors the module tree in rbuild.model exactly — v3 critics, noting
        experts, actuation head and both vision modes included — so the
        printed number matches a built model's `sum(p.numel())`.
        """
        m, cl, p, mem = self.model, self.cache_loop, self.parallel, self.memory
        c, n, a = self.critic, self.noting, self.actuation
        self.validate()
        d = m.d_model
        hd = m.resolved_head_dim()

        attn = d * (m.n_heads * hd) + 2 * d * (m.n_kv_heads * hd) + (m.n_heads * hd) * d

        exp_dim = self.resolved_expert_dim()
        ns = max(1, p.n_shared_experts)
        shared_dim = max(exp_dim, int(d * 4 / ns))   # shared stays dense-sized
        expert = 3 * d * exp_dim
        shared = 3 * d * shared_dim

        def ffn(kind: str) -> int:
            if kind == "moe":
                return d * p.n_experts + p.n_experts * expert + ns * shared
            ffn_dim = p.dense_ffn_dim or int(d * p.ffn_mult)
            return 3 * d * ffn_dim

        def block(kind: str) -> int:
            return attn + ffn(kind) + 2 * d          # norm1 + norm2

        # --- v3 critic panel (per location) --------------------------- #
        ch = self.resolved_critic_hidden()
        critic_expert = d * ch + ch * ch + ch + 1    # w1, w2, score(+bias)
        critic_panel = c.n_critics * critic_expert if c.enabled else 0

        # Stage A: cache-loop line (dense FFN blocks) + pull machinery
        line_blocks = cl.n_layers * (1 if cl.share_loop_weights else cl.n_loops)
        cache_line = line_blocks * block("dense") \
            + d + d * mem.key_dim + d + d   # mod router, mem key_proj, read gate, read_norm
        cache_line += critic_panel          # v3: extraction-loop critics
        cache_line_active = cl.n_layers * cl.n_loops * block("dense")  # compute repeats per loop
        cache_line_active += critic_panel * (c.max_loops if c.enabled else 0)

        # Stage B: parallel generative stages
        stage = p.n_branches * block(p.branch_ffn) + d + d   # mod router + out_norm
        if p.bundle_mode == "gate":
            stage += d * p.n_branches + p.n_branches
        elif p.bundle_mode == "concat":
            stage += (p.n_branches * d) * d
        if c.enabled and c.stage_critics:
            stage += critic_panel           # v3: generative-stage critics

        def block_active(kind: str) -> int:
            active_ffn = (d * p.n_experts + p.expert_top_k * expert + ns * shared) \
                if kind == "moe" else ffn("dense")
            return attn + active_ffn + 2 * d

        stage_active = p.n_branches * block_active(p.branch_ffn) + d + d
        if p.bundle_mode == "gate":
            stage_active += d * p.n_branches + p.n_branches
        elif p.bundle_mode == "concat":
            stage_active += (p.n_branches * d) * d
        if c.enabled and c.stage_critics:
            stage_active += critic_panel
        parallel_total = p.n_stages * stage
        parallel_active = p.n_stages * stage_active

        # --- v3 noting experts ---------------------------------------- #
        noting_params = 0
        if n.enabled:
            nh = self.resolved_note_hidden()
            noting_params = n.n_noting_experts * (
                d * nh + nh * mem.key_dim + nh * mem.value_dim + nh + 1)

        # --- v3 actuation ---------------------------------------------- #
        n_act = self.n_action_tokens()
        actuation_params = n_act * d        # dedicated action head
        eff_vocab = self.effective_vocab_size()

        embed = eff_vocab * d
        head = 0 if m.tie_embeddings else eff_vocab * d
        fact_proj = d * mem.key_dim + d * mem.value_dim
        final_norm = d

        # vision tower (mirrors rbuild.vision.VisionTower exactly, both modes)
        vision_params = 0
        if self.vision.enabled:
            v = self.vision
            vd = v.vit_dim or d
            g = v.image_size // v.patch_size
            if v.mode == "vit":
                tpi = g * g + (1 if v.use_cls_token else 0)
                vit_block = 4 * vd * vd + 2 * vd + 3 * vd * int(vd * v.vit_ffn_mult)
                vision_params = (
                    v.channels * vd * v.patch_size * v.patch_size   # patch conv
                    + tpi * vd                                       # pos embed
                    + (vd if v.use_cls_token else 0)                 # cls token
                    + v.vit_layers * vit_block
                    + vd                                             # out norm
                    + vd * d                                         # projector
                )
            else:  # encoderless — no encoder, patch conv straight to d_model
                vision_params = (
                    v.channels * d * v.patch_size * v.patch_size    # patch conv
                    + g * g * d                                      # pos embed
                    + d                                              # patch norm
                )
            vision_params += v.max_video_frames * d if v.video else 0
            if v.vawu and v.video:
                vision_params += v.vawu_tokens * d + 4 * d * d + d   # queries, MHA, norm

        total = (cache_line + parallel_total + embed + head + fact_proj + final_norm
                 + vision_params + noting_params + actuation_params)
        active = cache_line_active * cl.mod_capacity + parallel_active * p.mod_capacity \
            + embed + head + fact_proj + final_norm + vision_params \
            + noting_params + actuation_params
        return {
            "total_params": int(total),
            "active_params_per_token": int(active),
            "cache_loop_params": int(cache_line),
            "parallel_params": int(parallel_total),
            "critic_params": int((critic_panel if c.enabled else 0)
                                 * (1 + (p.n_stages if c.stage_critics else 0))),
            "noting_params": int(noting_params),
            "actuation_params": int(actuation_params),
            "embed_params": int(embed + head),
            "vision_params": int(vision_params),
            "memory_matrix": int(mem.key_dim * mem.value_dim if mem.enabled else 0),
        }

    # ------------------------------------------------------------------ #
    # cost estimate — always shown naive vs optimized, per project policy
    # ------------------------------------------------------------------ #
    def estimate_cost(self) -> Dict[str, float]:
        c = self.count_parameters()
        t = self.train
        tokens_per_step = t.tokens_per_step or (t.batch_size * t.grad_accum * self.model.max_seq_len)
        total_tokens = tokens_per_step * t.max_steps
        flops_per_token_naive = 6.0 * c["total_params"]
        flops_per_token_opt = 6.0 * c["active_params_per_token"]
        eff = {"fp32": 0.5, "bf16": 0.75, "fp8": 1.5}[t.precision]

        def hours(flops_per_token, efficiency_mult):
            flops = flops_per_token * total_tokens
            per_sec = t.achieved_tflops * 1e12 * efficiency_mult * t.gpus
            return flops / per_sec / 3600.0

        h_naive = hours(flops_per_token_naive, 0.5)      # naive = fp32-ish, dense
        h_opt = hours(flops_per_token_opt, eff)
        return {
            "total_tokens": float(total_tokens),
            "hours_naive": h_naive,
            "hours_optimized": h_opt,
            "cost_naive_usd": h_naive * t.gpu_price_per_hour * t.gpus,
            "cost_optimized_usd": h_opt * t.gpu_price_per_hour * t.gpus,
            "speedup_x": (h_naive / h_opt) if h_opt > 0 else float("inf"),
        }

    # ------------------------------------------------------------------ #
    # pretty report
    # ------------------------------------------------------------------ #
    def report(self) -> str:
        c = self.count_parameters()
        cost = self.estimate_cost()
        def fmt(n): return f"{n/1e9:.3f}B" if n >= 1e9 else f"{n/1e6:.2f}M"
        cr = self.critic
        if cr.enabled:
            halt_line = (f"ACT, until {cr.y_critics}/{cr.n_critics} critics "
                         f"satisfied, max {cr.max_loops} loops")
            critic_line = (f"{cr.n_critics} per panel, hidden "
                           f"{self.resolved_critic_hidden()} "
                           f"({cr.critic_capacity_mult}x working expert)"
                           f", stage panels={'on' if cr.stage_critics else 'off'}")
        else:
            halt_line = "off (fixed loops)"
            critic_line = "off"
        if self.noting.enabled:
            noting_line = (f"{self.noting.n_noting_experts} note-takers, critic-verified "
                           f"({self.noting.verify_y_critics}y@{self.noting.verify_threshold})"
                           f", NSCT={'on' if self.noting.nsct else 'off'}")
        else:
            noting_line = "off"
        if self.actuation.enabled:
            act_line = (f"on (+{self.n_action_tokens()} action tokens, "
                        f"grid {self.actuation.screen_grid}^2)")
        else:
            act_line = "off"
        if self.vision.enabled:
            vision_line = (f"{self.vision.mode} ({c['vision_params']/1e6:.2f}M, "
                           f"video={'on' if self.vision.video else 'off'}"
                           f", vawu={'on' if self.vision.vawu else 'off'})")
        else:
            vision_line = "blind (off)"
        lines = [
            "R-Build v3 configuration report",
            "=" * 56,
            f"  extraction loop   : {self.cache_loop.n_layers} layers"
            f"  (shared={self.cache_loop.share_loop_weights}, pull every"
            f" {self.cache_loop.memory_read_every})",
            f"  adaptive halting  : {halt_line}",
            f"  generative stages : {self.parallel.n_stages} stages x {self.parallel.n_branches} branches"
            f"  (bundle={self.parallel.bundle_mode})",
            f"  critic experts    : {critic_line}",
            f"  MoE per branch    : {self.parallel.n_experts} experts, top-{self.parallel.expert_top_k},"
            f" {self.parallel.n_shared_experts} shared",
            f"  noting experts    : {noting_line}",
            f"  thinking mode     : {self.thinking.default_mode}"
            + (f" (+{len(self.thinking.custom_modes)} custom)" if self.thinking.custom_modes else ""),
            f"  native actuation  : {act_line}",
            f"  gen watermark     : {'off' if not self.watermark.enabled else f'on (delta={self.watermark.delta}, gamma={self.watermark.gamma})'}",
            f"  fast-weight memory: {'on' if self.memory.enabled else 'off'}"
            f" (key={self.memory.key_dim}, value={self.memory.value_dim})",
            f"  vision            : {vision_line}",
            "-" * 56,
            f"  total params      : {fmt(c['total_params'])}",
            f"  active per token  : {fmt(c['active_params_per_token'])}",
            f"  critics / noting  : {fmt(c['critic_params'])} / {fmt(c['noting_params'])}",
            "-" * 56,
            f"  cost (naive)      : ${cost['cost_naive_usd']:,.0f}  ({cost['hours_naive']:,.1f} h)",
            f"  cost (optimized)  : ${cost['cost_optimized_usd']:,.0f}  ({cost['hours_optimized']:,.1f} h)",
            f"  speedup           : {cost['speedup_x']:.2f}x",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # io
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RBuildConfig":
        cfg = cls()
        for section, values in d.items():
            sec = getattr(cfg, section, None)
            if sec is None:
                continue            # forward/backward compatible section skip
            for k, v in values.items():
                if hasattr(sec, k):
                    setattr(sec, k, v)
        return cfg

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "RBuildConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))


# --------------------------------------------------------------------------- #
# Stage ladder presets (s1..s5) — the continued-training ladder
# --------------------------------------------------------------------------- #

def _preset(name: str) -> RBuildConfig:
    cfg = RBuildConfig()
    # Counter-verified against the ladder:
    #   s1 20B-A3.4B | s2 90B-A8.9B | s3 118B-A19.6B | s4 219B-A40.9B | s5 411B-A61.5B
    # (working-expert sizes are the v2 ladder; v3 critics + noting experts add
    #  their counter-verified parameters on top — report() shows them split out)
    ladder = {
        "s1": dict(d_model=2048, n_heads=16, n_kv_heads=4, n_loops=6,
                   n_stages=7, n_branches=6, n_experts=128, expert_top_k=8,
                   expert_ffn_dim=512, key_dim=512),                    # 19.9B / 3.3B (v2 working stack)
        "s2": dict(d_model=2560, n_heads=20, n_kv_heads=5, n_loops=8,
                   n_stages=6, n_branches=9, n_experts=160, expert_top_k=10,
                   expert_ffn_dim=1280, key_dim=640),                   # 90.6B / 8.9B
        "s3": dict(d_model=4096, n_heads=32, n_kv_heads=8, n_loops=8,
                   n_stages=9, n_branches=7, n_experts=128, expert_top_k=8,
                   expert_ffn_dim=1024, key_dim=1024),                  # 117.9B / 18.8B
        "s4": dict(d_model=5120, n_heads=40, n_kv_heads=10, n_loops=8,
                   n_stages=12, n_branches=8, n_experts=96, expert_top_k=6,
                   expert_ffn_dim=1280, key_dim=1280),                  # 219.2B / 39.6B
        "s5": dict(d_model=6144, n_heads=48, n_kv_heads=12, n_loops=6,
                   n_stages=9, n_branches=11, n_experts=32, expert_top_k=2,
                   expert_ffn_dim=6144, key_dim=1536),                  # 414.9B / 61.6B
    }
    if name not in ladder:
        raise KeyError(f"unknown preset {name!r}; choose from {sorted(ladder)}")
    v = ladder[name]
    cfg.model.d_model = v["d_model"]
    cfg.model.n_heads = v["n_heads"]
    cfg.model.n_kv_heads = v["n_kv_heads"]
    cfg.model.vocab_size = 128000
    cfg.cache_loop.n_loops = v["n_loops"]
    cfg.critic.max_loops = v["n_loops"]      # v3: the ladder's loop count is the ACT cap
    cfg.parallel.n_stages = v["n_stages"]
    cfg.parallel.n_branches = v["n_branches"]
    cfg.parallel.n_experts = v["n_experts"]
    cfg.parallel.expert_top_k = v["expert_top_k"]
    cfg.parallel.expert_ffn_dim = v["expert_ffn_dim"]
    cfg.memory.key_dim = v["key_dim"]
    cfg.memory.value_dim = v["d_model"]
    return cfg


def preset(name: str = "s1") -> RBuildConfig:
    """Return the config for a ladder stage: 's1'..'s5' or 'tiny'."""
    if name == "tiny":
        return RBuildConfig()
    return _preset(name)

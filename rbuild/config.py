"""
R-Build v2 — fully user-modifiable configuration.

Every value in the engine lives here and can be changed:
  - programmatically:      cfg = RBuildConfig(); cfg.parallel.n_branches = 8
  - interactively:         rbuild.interactive.launch()  (widgets in Colab/Jupyter,
                           console prompts elsewhere)
  - from the CLI:          python -m rbuild.interactive

The config is organized in five sections:

  ModelConfig      — sizes of the network itself
  CacheLoopConfig  — the sequential "single line" stage that loops to pull
                     the fast-weight memory cache
  ParallelConfig   — the parallel stages whose branches bundle outputs and
                     push them to the next parallel stage
  MemoryConfig     — the gradient-free fast-weight delta-rule memory
  VisionConfig     — ViT tower for image/video soft tokens (blind by default)
  TrainConfig      — Muon / WSD / precision / batching / cost model

Validation and derived counters (parameters, active parameters, estimated
training cost with the naive-vs-optimized comparison) run on demand via
`cfg.report()` and automatically inside the interactive panel.
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

    A shared line of `n_layers` blocks is applied `n_loops` times. Every loop
    iteration performs a delta-rule *read* against the fast-weight memory
    cache and injects the retrieved value into the residual stream, so the
    state is progressively refined by what the cache holds.

    Set `share_loop_weights=True` (default) so the loop re-uses the same
    weights — this is what makes the stage a true loop rather than a stack.
    """
    n_layers: int = 2                # blocks in the single line
    n_loops: int = 4                 # how many times the line loops
    share_loop_weights: bool = True  # reuse weights across loops
    memory_read_every: int = 1       # pull cache every k-th loop iteration
    use_mod_routing: bool = True     # MoD: only top-p tokens hit each block
    mod_capacity: float = 0.5        # top fraction of tokens routed per block


@dataclass
class ParallelConfig:
    """
    Stage B — the parallel stages.

    Each stage owns `n_branches` layer-branches that run in parallel over the
    same input. Their outputs are *bundled* by a learned gate (or mean /
    concat+project, see `bundle_mode`) and the bundle is pushed to the next
    stage. During generation each stage's bundle is what produces the hidden
    state that the next parallel stage consumes.
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
    VL support (v2.1). Default is *blind*: enabled=False means the vision
    tower is never built and the model is bit-identical to the text-only
    v2.0 path. Set enabled=True and image_token_id to go multimodal.

    Images (or sampled video frames) are encoded by a ViT tower, projected
    to d_model, and spliced into the token stream at `image_token_id`
    placeholder positions — downstream stages just see more tokens.
    """
    enabled: bool = False            # False = blind (text-only), zero overhead
    image_token_id: Optional[int] = None  # reserved placeholder id in the vocab
    image_size: int = 224
    patch_size: int = 14
    channels: int = 3
    vit_layers: int = 6
    vit_dim: Optional[int] = None    # None -> d_model
    vit_heads: int = 12
    vit_ffn_mult: float = 4.0
    use_cls_token: bool = False
    freeze_vision: bool = False      # train projector only (cheap VL bootstrap)
    # video (s4-style whole-video understanding)
    video: bool = True               # allow (B, frames, C, H, W) inputs
    max_video_frames: int = 16       # learned frame-position table size


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
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # ------------------------------------------------------------------ #
    # validation
    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        m, cl, p, mem = self.model, self.cache_loop, self.parallel, self.memory
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
        v = self.vision
        if v.enabled:
            assert v.image_token_id is not None, \
                "vision.enabled=True requires an image_token_id placeholder"
            assert v.image_size % v.patch_size == 0, "image_size must divide patch_size"
            vd = v.vit_dim or m.d_model
            assert vd % v.vit_heads == 0, "vit_dim must divide vit_heads"

    # ------------------------------------------------------------------ #
    # parameter counter (naive vs optimized stack)
    # ------------------------------------------------------------------ #
    def count_parameters(self) -> Dict[str, int]:
        """
        Static count of total / active (per-token) parameters. The counting
        mirrors the module tree in rbuild.model exactly, so the printed
        number matches a built model's `sum(p.numel())` — the counter is
        the verification tool, not an approximation.
        """
        m, cl, p, mem = self.model, self.cache_loop, self.parallel, self.memory
        self.validate()
        d = m.d_model
        hd = m.resolved_head_dim()

        attn = d * (m.n_heads * hd) + 2 * d * (m.n_kv_heads * hd) + (m.n_heads * hd) * d

        exp_dim = p.expert_ffn_dim or max(8, int(d * p.ffn_mult / p.n_experts))
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

        # Stage A: cache-loop line (dense FFN blocks) + pull machinery
        line_blocks = cl.n_layers * (1 if cl.share_loop_weights else cl.n_loops)
        cache_line = line_blocks * block("dense") \
            + d + d * mem.key_dim + d + d   # mod router, mem key_proj, read gate, read_norm
        cache_line_active = cl.n_layers * cl.n_loops * block("dense")  # compute repeats per loop

        # Stage B: parallel stages
        stage = p.n_branches * block(p.branch_ffn) + d + d   # mod router + out_norm
        if p.bundle_mode == "gate":
            stage += d * p.n_branches + p.n_branches
        elif p.bundle_mode == "concat":
            stage += (p.n_branches * d) * d

        def block_active(kind: str) -> int:
            active_ffn = (d * p.n_experts + p.expert_top_k * expert + ns * shared) \
                if kind == "moe" else ffn("dense")
            return attn + active_ffn + 2 * d

        stage_active = p.n_branches * block_active(p.branch_ffn) + d + d
        if p.bundle_mode == "gate":
            stage_active += d * p.n_branches + p.n_branches
        elif p.bundle_mode == "concat":
            stage_active += (p.n_branches * d) * d
        parallel_total = p.n_stages * stage
        parallel_active = p.n_stages * stage_active

        embed = m.vocab_size * d
        head = 0 if m.tie_embeddings else m.vocab_size * d
        fact_proj = d * mem.key_dim + d * mem.value_dim
        final_norm = d

        # vision tower (mirrors rbuild.vision.VisionTower exactly)
        vision_params = 0
        if self.vision.enabled:
            v = self.vision
            vd = v.vit_dim or d
            g = v.image_size // v.patch_size
            tpi = g * g + (1 if v.use_cls_token else 0)
            vit_block = 4 * vd * vd + 2 * vd + 3 * vd * int(vd * v.vit_ffn_mult)
            vision_params = (
                v.channels * vd * v.patch_size * v.patch_size   # patch conv
                + tpi * vd                                       # pos embed
                + (vd if v.use_cls_token else 0)                 # cls token
                + v.vit_layers * vit_block
                + vd                                             # out norm
                + vd * d                                         # projector
                + (v.max_video_frames * d if v.video else 0)     # frame embed
            )

        total = cache_line + parallel_total + embed + head + fact_proj + final_norm + vision_params
        active = cache_line_active * cl.mod_capacity + parallel_active * p.mod_capacity \
            + embed + head + fact_proj + final_norm + vision_params
        return {
            "total_params": int(total),
            "active_params_per_token": int(active),
            "cache_loop_params": int(cache_line),
            "parallel_params": int(parallel_total),
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
        lines = [
            "R-Build configuration report",
            "=" * 56,
            f"  cache-loop line   : {self.cache_loop.n_layers} layers x {self.cache_loop.n_loops} loops"
            f"  (shared={self.cache_loop.share_loop_weights}, pull every {self.cache_loop.memory_read_every})",
            f"  parallel stages   : {self.parallel.n_stages} stages x {self.parallel.n_branches} branches"
            f"  (bundle={self.parallel.bundle_mode})",
            f"  MoE per branch    : {self.parallel.n_experts} experts, top-{self.parallel.expert_top_k},"
            f" {self.parallel.n_shared_experts} shared",
            f"  fast-weight memory: {'on' if self.memory.enabled else 'off'}"
            f" (key={self.memory.key_dim}, value={self.memory.value_dim})",
            f"  vision            : {'blind (off)' if not self.vision.enabled else 'on'}"
            + (f" ({c['vision_params']/1e6:.2f}M tower, {self.vision.vit_layers} ViT layers"
               f", video={'on' if self.vision.video else 'off'})" if self.vision.enabled else ""),
            "-" * 56,
            f"  total params      : {fmt(c['total_params'])}",
            f"  active per token  : {fmt(c['active_params_per_token'])}",
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
            sec = getattr(cfg, section)
            for k, v in values.items():
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
    # (all within ~4% under the v2 loop-cache + parallel-bundle architecture;
    #  s3 deliberately goes wider-per-token, not deeper)
    ladder = {
        "s1": dict(d_model=2048, n_heads=16, n_kv_heads=4, n_loops=6,
                   n_stages=7, n_branches=6, n_experts=128, expert_top_k=8,
                   expert_ffn_dim=512, key_dim=512),                    # 19.9B / 3.3B
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

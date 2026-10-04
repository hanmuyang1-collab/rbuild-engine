"""
R-Build v3 architecture.

v2 gave us the two-stage skeleton:
  Stage A — CacheLoopLine: a single line of blocks, looped, pulling the
            fast-weight cache each pass.
  Stage B — ParallelBundleStage: parallel branches whose outputs bundle
            and push to the next stage.

v3 makes the skeleton *self-governing*:

  1. Critic-gated extraction (ACT-style halting). The cache loop becomes
     the *extraction loop*: after each iteration a panel of X parallel
     critic experts scores every token; tokens halt as their cumulative
     satisfaction crosses 1 (ACT), and the loop early-exits once Y critics
     are satisfied on average. A ponder cost (loops + remainders) joins
     the training loss.

  2. Critic experts on the generative layers. Every parallel stage carries
     X critics with MORE capacity than its working experts. They judge the
     bundle and are the verifiers of the self-training pipeline.

  3. Noting experts + non-separate self-training. Note-takers watch the
     final hidden states during normal use (including generation), critics
     verify, verified notes enter fast-weight memory gradient-free AND queue
     in a low-RAM CPU buffer for consolidation into the slow weights.
     Running and learning are the same pass.

  4. Thinking modes: `model.thinking_mode.<mode>(<value>)` retunes loops,
     Y-critics, thresholds and sampling live; users mint their own modes.

  5. Native actuation (optional): reserved action tokens + a dedicated head
     — the model clicks/scrolls/types by generating a token.

  6. Vision (optional): v2.1 ViT mode, or v3 encoderless mode (no vision
     encoder — patches are projected straight into d_model), with optional
     VaWU whole-video summary tokens.

Everything v3 can be switched off in the config; with critic/noting/
actuation/vision disabled the architecture is bit-identical to v2.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .config import RBuildConfig
from .layers.attention import CausalAttention, MoDRouter, RMSNorm
from .layers.moe import build_ffn
from .memory import FastWeightMemory
from .vision import VisionTower
from .critics import CriticPanel, ACTHalting
from .noting import NotingExperts, SelfLearner
from .thinking import ThinkingModes
from .actuation import ActionCodec, ActuationHead


# --------------------------------------------------------------------------- #
# building blocks
# --------------------------------------------------------------------------- #

class Block(nn.Module):
    """Attention + FFN block (pre-norm). The atom of both stages."""

    def __init__(self, cfg: RBuildConfig, ffn_kind: str):
        super().__init__()
        m = cfg.model
        self.norm1 = RMSNorm(m.d_model, m.rmsnorm_eps)
        self.attn = CausalAttention(m.d_model, m.n_heads, m.n_kv_heads,
                                    m.head_dim, m.rope_theta, m.max_seq_len)
        self.norm2 = RMSNorm(m.d_model, m.rmsnorm_eps)
        self.ffn = build_ffn(
            ffn_kind, m.d_model,
            n_experts=cfg.parallel.n_experts, top_k=cfg.parallel.expert_top_k,
            expert_dim=cfg.parallel.expert_ffn_dim,
            n_shared=cfg.parallel.n_shared_experts,
            dense_dim=cfg.parallel.dense_ffn_dim, ffn_mult=cfg.parallel.ffn_mult,
        )

    def forward(self, x, kv_cache=None, start_pos=0, use_cache=False):
        a, new_cache = self.attn(self.norm1(x), kv_cache=kv_cache,
                                 start_pos=start_pos, use_cache=use_cache)
        x = x + a
        x = x + self.ffn(self.norm2(x))
        return x, new_cache


class _TokenSubsetBlock(nn.Module):
    """Adapter so MoDRouter can call a full Block on gathered tokens."""

    def __init__(self, block: Block):
        super().__init__()
        self.block = block

    def forward(self, x, **_):
        # routed token subset: treat as independent length-1 sequences
        out, _ = self.block(x, kv_cache=None, start_pos=0, use_cache=False)
        return out


# --------------------------------------------------------------------------- #
# Stage A — the extraction loop (cache-pulling, critic-gated in v3)
# --------------------------------------------------------------------------- #

class CacheLoopLine(nn.Module):
    def __init__(self, cfg: RBuildConfig):
        super().__init__()
        self.cfg = cfg
        cl = cfg.cache_loop
        self.n_loops = cl.n_loops
        self.read_every = cl.memory_read_every
        if cl.share_loop_weights:
            # one line of weights, reused every loop — a true loop
            self.line = nn.ModuleList(Block(cfg, ffn_kind="dense")
                                      for _ in range(cl.n_layers))
            self.loop_line = None
        else:
            # each loop iteration gets its own copy of the line
            self.line = None
            self.loop_line = nn.ModuleList(
                nn.ModuleList(Block(cfg, ffn_kind="dense") for _ in range(cl.n_loops))
                for _ in range(cl.n_loops)
            )
        self.mod = MoDRouter(cfg.model.d_model, cl.mod_capacity) if cl.use_mod_routing else None
        # cache pull projections + gate
        self.mem_key_proj = nn.Linear(cfg.model.d_model, cfg.memory.key_dim, bias=False)
        self.read_gate = nn.Parameter(torch.full((cfg.model.d_model,), cfg.memory.read_gate_init))
        self.read_norm = RMSNorm(cfg.model.d_model, cfg.model.rmsnorm_eps)
        # v3: critic panel gating the loop (ACT-style halting)
        c = cfg.critic
        self.critics = CriticPanel(cfg.model.d_model, cfg.resolved_critic_hidden(),
                                   c.n_critics, c.threshold) if c.enabled else None
        self.last_halting: Optional[dict] = None

    def _pull_cache(self, x: torch.Tensor, memory: FastWeightMemory) -> torch.Tensor:
        """Delta-rule read against the fast-weight cache, gated injection."""
        keys = self.mem_key_proj(x)                      # (B, T, kd)
        retrieved = memory.read(keys)                    # (B, T, vd==d_model)
        return x + torch.tanh(self.read_gate) * retrieved

    def _run_blocks(self, x: torch.Tensor, loop: int) -> torch.Tensor:
        blocks = self.line if self.line is not None else self.loop_line[loop]
        for blk in blocks:
            if self.mod is not None:
                x = self.mod(x, _TokenSubsetBlock(blk))
            else:
                x, _ = blk(x)
        return x

    def forward(self, x: torch.Tensor, memory: Optional[FastWeightMemory],
                max_loops: Optional[int] = None,
                y_critics: Optional[int] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns (hidden, ponder_cost_or_None). With critics enabled the loop
        is adaptive; without them it runs exactly n_loops iterations (v2).
        """
        if self.critics is None:
            n = self.n_loops
            for loop in range(n):
                x = self._run_blocks(x, loop % self.n_loops)
                if memory is not None and (loop + 1) % self.read_every == 0:
                    x = self._pull_cache(self.read_norm(x), memory)
            self.last_halting = None
            return x, None

        # ---- v3: ACT-style critic-gated extraction ----
        c = self.cfg.critic
        cap = max_loops or c.max_loops
        halt = ACTHalting(c, x.shape[0], x.shape[1], x.device,
                          max_loops=cap, y_critics=y_critics)
        frozen = x
        prev_running = halt.running.clone()
        for loop in range(cap):
            x_new = self._run_blocks(x, loop % self.n_loops)
            if memory is not None and (loop + 1) % self.read_every == 0:
                x_new = self._pull_cache(self.read_norm(x_new), memory)
            done = halt.step(x_new, self.critics)
            # halted tokens keep their final state; running tokens continue
            just_halted = prev_running & ~halt.running
            frozen = torch.where(just_halted.unsqueeze(-1), x_new, frozen)
            x = torch.where(halt.running.unsqueeze(-1), x_new, frozen)
            prev_running = halt.running.clone()
            if done:
                break
        self.last_halting = halt.stats()
        return x, halt.ponder_cost()


# --------------------------------------------------------------------------- #
# Stage B — parallel generative stages (with critic panels in v3)
# --------------------------------------------------------------------------- #

class ParallelBundleStage(nn.Module):
    def __init__(self, cfg: RBuildConfig):
        super().__init__()
        p = cfg.parallel
        self.bundle_mode = p.bundle_mode
        self.branches = nn.ModuleList(Block(cfg, ffn_kind=p.branch_ffn)
                                      for _ in range(p.n_branches))
        self.mod = MoDRouter(cfg.model.d_model, p.mod_capacity) if p.use_mod_routing else None
        d = cfg.model.d_model
        if p.bundle_mode == "gate":
            self.gate = nn.Linear(d, p.n_branches)         # per-position branch weights
        elif p.bundle_mode == "concat":
            self.proj = nn.Linear(p.n_branches * d, d, bias=False)
        self.out_norm = RMSNorm(d, cfg.model.rmsnorm_eps)
        # v3: X parallel critic experts, wider than the working experts
        c = cfg.critic
        self.critics = CriticPanel(d, cfg.resolved_critic_hidden(),
                                   c.n_critics, c.threshold) \
            if (c.enabled and c.stage_critics) else None

    def _bundle(self, outs: List[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        if self.bundle_mode == "gate":
            w = F.softmax(self.gate(x), dim=-1)            # (B, T, n_branches)
            stacked = torch.stack(outs, dim=-1)            # (B, T, D, n_branches)
            return (stacked * w.unsqueeze(2)).sum(-1)
        if self.bundle_mode == "concat":
            return self.proj(torch.cat(outs, dim=-1))
        return torch.stack(outs, 0).mean(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        for branch in self.branches:
            if self.mod is not None:
                outs.append(self.mod(x, _TokenSubsetBlock(branch)))
            else:
                outs.append(branch(x)[0])
        return x + self.out_norm(self._bundle(outs, x))

    def critic_verdict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(satisfaction, n_satisfied) of this stage's critics on x."""
        assert self.critics is not None, "stage critics disabled"
        return self.critics(x)


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #

class RBuildModel(nn.Module):
    def __init__(self, cfg: RBuildConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        m = cfg.model
        eff_vocab = cfg.effective_vocab_size()
        self.embed = nn.Embedding(eff_vocab, m.d_model)
        self.cache_loop = CacheLoopLine(cfg)
        self.stages = nn.ModuleList(ParallelBundleStage(cfg)
                                    for _ in range(cfg.parallel.n_stages))
        self.final_norm = RMSNorm(m.d_model, m.rmsnorm_eps)
        self.lm_head = nn.Linear(m.d_model, eff_vocab, bias=False)
        if m.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        self.memory = FastWeightMemory(cfg.memory.key_dim, cfg.memory.value_dim,
                                       cfg.memory.write_lr, cfg.memory.decay,
                                       cfg.memory.max_facts) if cfg.memory.enabled else None
        # projections for text fact writes (gradient-free path)
        self.fact_key_proj = nn.Linear(m.d_model, cfg.memory.key_dim, bias=False)
        self.fact_value_proj = nn.Linear(m.d_model, cfg.memory.value_dim, bias=False)

        self.drop = nn.Dropout(m.dropout)

        # vision tower — not built at all in blind mode
        self.vision = VisionTower(cfg) if cfg.vision.enabled else None
        if self.vision is not None and cfg.vision.freeze_vision:
            for name, p_ in self.vision.named_parameters():
                if "projector" not in name:
                    p_.requires_grad = False

        # v3: noting experts + non-separate self-learner
        self.noting_experts = NotingExperts(m.d_model, cfg.memory.key_dim,
                                            cfg.memory.value_dim,
                                            cfg.noting.n_noting_experts,
                                            cfg.resolved_note_hidden()) \
            if cfg.noting.enabled else None
        self.self_learner = SelfLearner(self) if cfg.noting.enabled else None

        # v3: native actuation — action codec + dedicated action head
        self.action_codec = ActionCodec(m.vocab_size, cfg.actuation.screen_grid,
                                        cfg.actuation.scroll_steps) \
            if cfg.actuation.enabled else None
        self.action_head = ActuationHead(m.d_model, cfg.n_action_tokens()) \
            if cfg.actuation.enabled else None

        # v3: runtime knobs driven by thinking modes
        self._runtime_max_loops: Optional[int] = None
        self._runtime_y_critics: Optional[int] = None
        self._runtime_sampling: Optional[dict] = None
        self._runtime_self_observe: bool = cfg.noting.enabled and cfg.noting.nsct
        self._active_thinking_mode: Optional[str] = None
        self._watermarker_obj = None
        self.thinking_mode = ThinkingModes(self)
        if cfg.critic.enabled and cfg.thinking.default_mode:
            try:
                self.thinking_mode.apply(cfg.thinking.default_mode)
            except KeyError:
                pass
        # v3.1: NSCT (non-separate continuous training) is OPT-IN —
        # startup never self-trains unless noting.nsct=True, even if the
        # default thinking mode carries self_observe=True. Explicitly
        # applying a mode or calling set_nsct(True) turns it on later.
        if not cfg.noting.nsct:
            self._runtime_self_observe = False

        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    # ------------------------------------------------------------------ #
    def _splice_vision(self, input_ids: torch.Tensor,
                       images: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Replace `<image>` placeholder embeddings with vision soft tokens
        (ViT or encoderless mode; VaWU whole-video tokens included).
        Returns (inputs_embeds, image_mask) — mask marks vision positions so
        the loss can skip them.
        """
        emb = self.embed(input_ids.clamp_max(self.embed.num_embeddings - 1))
        if self.vision is None:
            return emb, None
        if images is None:
            mask = input_ids == self.cfg.vision.image_token_id
            if mask.any():
                raise ValueError("input contains image placeholders but no images were passed")
            return emb, None
        img_tokens = self.vision(images)                       # (B, K, D)
        B, K, D = img_tokens.shape
        mask = input_ids == self.cfg.vision.image_token_id     # (B, T)
        counts = mask.sum(1)
        if not (counts == K).all():
            raise ValueError(
                f"each sample needs exactly {K} image placeholders "
                f"(got {counts.tolist()}); one placeholder per vision soft token"
                + (f" (note: VaWU adds {self.cfg.vision.vawu_tokens} whole-video tokens)"
                   if (self.cfg.vision.vawu and images.shape[1] > 1) else ""))
        emb = emb.clone()
        emb[mask] = img_tokens.reshape(-1, D).to(emb.dtype)
        return emb, mask

    # ------------------------------------------------------------------ #
    def _run_block(self, blk, x):
        """Run a block, with optional activation checkpointing in training."""
        if self.training and self.cfg.train.grad_checkpoint and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(
                lambda t: blk(t)[0], x, use_reentrant=False)
        return blk(x)[0]

    # ------------------------------------------------------------------ #
    def critics_verify(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The generative critics' verdict on a hidden state — used by the
        self-learner to approve/reject notes. Uses the final stage's panel
        (falls back to the extraction-loop panel; without any critics,
        everything passes with a neutral verdict).
        """
        for stage in reversed(self.stages):
            if stage.critics is not None:
                return stage.critic_verdict(hidden)
        if self.cache_loop.critics is not None:
            return self.cache_loop.critics(hidden)
        ones = torch.ones(hidden.shape[:2], device=hidden.device)
        return ones, ones.to(torch.long) * 10**6

    # ------------------------------------------------------------------ #
    def forward(self, input_ids: torch.Tensor,
                targets: Optional[torch.Tensor] = None,
                images: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x, image_mask = self._splice_vision(input_ids, images)
        x = self.drop(x)
        x, ponder = self.cache_loop(x, self.memory,
                                    max_loops=self._runtime_max_loops,
                                    y_critics=self._runtime_y_critics)
        for stage in self.stages:
            if self.training and self.cfg.train.grad_checkpoint and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(stage, x, use_reentrant=False)
            else:
                x = stage(x)
        x = self.final_norm(x)

        # v3: non-separate self-training — note + verify + learn while running
        if (self.self_learner is not None and self._runtime_self_observe
                and (not self.training or self.cfg.noting.observe_in_training)):
            self._self_observe(x)

        logits = self.lm_head(x)
        # v3: action columns come from the dedicated actuation head
        if self.action_head is not None:
            n_text = self.cfg.model.vocab_size
            logits = torch.cat([logits[..., :n_text], self.action_head(x)], dim=-1)

        loss = None
        if targets is not None:
            if image_mask is not None:
                targets = targets.masked_fill(image_mask, -100)   # never predict vision positions
            loss = self._chunked_ce(logits, targets)
            loss = loss + 0.01 * self._moe_aux_loss()
            if ponder is not None:                                # v3: ACT halting loss
                loss = loss + self.cfg.critic.halt_loss_weight * ponder
        return logits, loss

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _self_observe(self, hidden: torch.Tensor) -> int:
        """Note-taking + critic verification on a finished forward pass."""
        if self.self_learner is None:
            return 0
        return self.self_learner.observe(hidden)

    def self_learn_stats(self) -> dict:
        if self.self_learner is None:
            return {"enabled": False}
        return {"enabled": True, **self.self_learner.stats()}

    # ------------------------------------------------------------------ #
    def set_nsct(self, on: bool = True) -> None:
        """
        The NSCT switch: toggle non-separate continuous training at runtime.
        NSCT = training and running SIMULTANEOUSLY, at low RAM — the model
        learns from what it observes while it serves, no separate phase.
        Off (the default) = pure inference, no notes taken, nothing learned.
        """
        self.cfg.noting.nsct = bool(on)
        self._runtime_self_observe = bool(on) and self.self_learner is not None

    def set_auto_train(self, on: bool = True) -> None:
        """Legacy alias for set_nsct() — kept for v3.1 scripts."""
        self.set_nsct(on)

    # ------------------------------------------------------------------ #
    def _chunked_ce(self, logits, targets):
        """Chunked cross-entropy over the token axis (memory saver)."""
        B, T, V = logits.shape
        if not (self.cfg.train.chunked_ce and self.training):
            return F.cross_entropy(logits.reshape(-1, V).float(), targets.reshape(-1),
                                   ignore_index=-100)
        chunk = max(1, self.cfg.train.ce_chunk_tokens)
        flat_l = logits.reshape(-1, V)
        flat_t = targets.reshape(-1)
        total, count = 0.0, 0
        for i in range(0, flat_l.shape[0], chunk):
            ls = F.cross_entropy(flat_l[i:i + chunk].float(), flat_t[i:i + chunk],
                                 reduction="sum", ignore_index=-100)
            total = total + ls
            count += int((flat_t[i:i + chunk] != -100).sum())   # skip ignored positions
        return total / max(1, count)

    def _moe_aux_loss(self):
        aux = torch.zeros((), device=next(self.parameters()).device)
        n = 0
        for mod in self.modules():
            if hasattr(mod, "last_aux_loss") and mod.training:
                aux = aux + mod.last_aux_loss
                n += 1
        return aux / max(1, n)

    # ------------------------------------------------------------------ #
    # gradient-free fact writes — "learn without training"
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def remember(self, token_ids: torch.Tensor, lr: Optional[float] = None) -> None:
        """
        Write tokenized content straight into the fast-weight cache.
        token_ids: (N,) or (1, N). No gradients, no optimizer step.
        """
        assert self.memory is not None, "memory is disabled in this config"
        ids = token_ids.view(-1).to(next(self.parameters()).device)
        emb = self.embed(ids).mean(0, keepdim=True)          # (1, D)
        key = self.fact_key_proj(emb)
        value = self.fact_value_proj(emb)
        self.memory.write(key, value, lr=lr)
        self.memory._fact_log.append({"n_tokens": int(ids.numel())})

    # ------------------------------------------------------------------ #
    def _watermarker(self):
        """Lazily built green-list watermarker (param-free)."""
        if self._watermarker_obj is None:
            from .watermark import GreenListWatermark
            self._watermarker_obj = GreenListWatermark(
                self.cfg.watermark, self.cfg.effective_vocab_size())
        return self._watermarker_obj

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 64,
                 temperature: Optional[float] = None, top_p: Optional[float] = None,
                 top_k: int = 0, eos_id: Optional[int] = None,
                 images: Optional[torch.Tensor] = None,
                 watermark: Optional[bool] = None) -> torch.Tensor:
        # thinking modes set the sampling defaults
        rt = self._runtime_sampling or {}
        temperature = temperature if temperature is not None else rt.get("temperature", 1.0)
        top_p = top_p if top_p is not None else rt.get("top_p", 0.9)
        wm = self.cfg.watermark.enabled if watermark is None else watermark
        self.eval()
        out = input_ids
        for _ in range(max_new_tokens):
            window = out[:, -self.cfg.model.max_seq_len:]
            logits, _ = self(window, images=images)
            nxt_logits = logits[:, -1, :].float() / max(1e-6, temperature)
            if wm:                                            # v3: watermark bias
                nxt_logits = self._watermarker().bias_logits(nxt_logits, out[:, -1])
            if top_k > 0:
                v, _ = torch.topk(nxt_logits, min(top_k, nxt_logits.shape[-1]))
                nxt_logits[nxt_logits < v[:, [-1]]] = -float("inf")
            if 0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(nxt_logits, descending=True)
                cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                nxt_logits.scatter_(1, sorted_idx, sorted_logits.masked_fill(remove, -float("inf")))
            probs = F.softmax(nxt_logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            out = torch.cat([out, nxt], dim=1)
            if eos_id is not None and (nxt == eos_id).all():
                break
        return out

    # ------------------------------------------------------------------ #
    def num_params(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        return {"total_actual": total, **self.cfg.count_parameters()}

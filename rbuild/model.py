"""
R-Build v2 architecture.

Stage A — CacheLoopLine ("the first set of layers in a single line, loops to
pull cache"):
    A single line of blocks. The hidden state runs through the line
    `n_loops` times; on every `memory_read_every`-th pass the line performs
    a delta-rule read against the fast-weight cache and injects the
    retrieved value through a learned gate. This is the stage that *pulls*
    the cache into the representation.

Stage B — ParallelBundleStage ("the multiple parallel layers ... generate
tokens and bundle to push to the next set of parallel layers"):
    `n_branches` parallel branches all consume the same input. Their outputs
    are bundled — learned gate, mean, or concat+project — and the bundle is
    pushed to the next parallel stage. During token generation the final
    bundle is what produces logits.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import RBuildConfig
from .layers.attention import CausalAttention, MoDRouter, RMSNorm
from .layers.moe import build_ffn
from .memory import FastWeightMemory


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
# Stage A — the cache-pulling loop
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
                nn.ModuleList(Block(cfg, ffn_kind="dense") for _ in range(cl.n_layers))
                for _ in range(cl.n_loops)
            )
        self.mod = MoDRouter(cfg.model.d_model, cl.mod_capacity) if cl.use_mod_routing else None
        # cache pull projections + gate
        self.mem_key_proj = nn.Linear(cfg.model.d_model, cfg.memory.key_dim, bias=False)
        self.read_gate = nn.Parameter(torch.full((cfg.model.d_model,), cfg.memory.read_gate_init))
        self.read_norm = RMSNorm(cfg.model.d_model, cfg.model.rmsnorm_eps)

    def _pull_cache(self, x: torch.Tensor, memory: FastWeightMemory) -> torch.Tensor:
        """Delta-rule read against the fast-weight cache, gated injection."""
        keys = self.mem_key_proj(x)                      # (B, T, kd)
        retrieved = memory.read(keys)                    # (B, T, vd==d_model)
        return x + torch.tanh(self.read_gate) * retrieved

    def forward(self, x: torch.Tensor, memory: FastWeightMemory) -> torch.Tensor:
        for loop in range(self.n_loops):
            blocks = self.line if self.line is not None else self.loop_line[loop]
            for blk in blocks:
                if self.mod is not None:
                    x = self.mod(x, _TokenSubsetBlock(blk))
                else:
                    x, _ = blk(x)
            if (loop + 1) % self.read_every == 0:
                x = self._pull_cache(self.read_norm(x), memory)
        return x


# --------------------------------------------------------------------------- #
# Stage B — parallel bundle stages
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


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #

class RBuildModel(nn.Module):
    def __init__(self, cfg: RBuildConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        m = cfg.model
        self.embed = nn.Embedding(m.vocab_size, m.d_model)
        self.cache_loop = CacheLoopLine(cfg)
        self.stages = nn.ModuleList(ParallelBundleStage(cfg)
                                    for _ in range(cfg.parallel.n_stages))
        self.final_norm = RMSNorm(m.d_model, m.rmsnorm_eps)
        self.lm_head = nn.Linear(m.d_model, m.vocab_size, bias=False)
        if m.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        self.memory = FastWeightMemory(cfg.memory.key_dim, cfg.memory.value_dim,
                                       cfg.memory.write_lr, cfg.memory.decay,
                                       cfg.memory.max_facts) if cfg.memory.enabled else None
        # projections for text fact writes (gradient-free path)
        self.fact_key_proj = nn.Linear(m.d_model, cfg.memory.key_dim, bias=False)
        self.fact_value_proj = nn.Linear(m.d_model, cfg.memory.value_dim, bias=False)

        self.drop = nn.Dropout(m.dropout)
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    # ------------------------------------------------------------------ #
    def forward(self, input_ids: torch.Tensor,
                targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.drop(self.embed(input_ids))
        if self.memory is not None:
            x = self.cache_loop(x, self.memory)
        else:
            # memory off: loop still runs, pulls are skipped
            for loop in range(self.cache_loop.n_loops):
                blocks = self.cache_loop.line if self.cache_loop.line is not None \
                    else self.cache_loop.loop_line[loop]
                for blk in blocks:
                    x, _ = blk(x)
        for stage in self.stages:
            x = stage(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = self._chunked_ce(logits, targets)
            loss = loss + 0.01 * self._moe_aux_loss()
        return logits, loss

    # ------------------------------------------------------------------ #
    def _chunked_ce(self, logits, targets):
        """Chunked cross-entropy over the token axis (memory saver)."""
        B, T, V = logits.shape
        if not (self.cfg.train.chunked_ce and self.training):
            return F.cross_entropy(logits.reshape(-1, V).float(), targets.reshape(-1))
        chunk = max(1, self.cfg.train.ce_chunk_tokens)
        flat_l = logits.reshape(-1, V)
        flat_t = targets.reshape(-1)
        total, count = 0.0, 0
        for i in range(0, flat_l.shape[0], chunk):
            ls = F.cross_entropy(flat_l[i:i + chunk].float(), flat_t[i:i + chunk], reduction="sum")
            total = total + ls
            count += flat_t[i:i + chunk].numel()
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
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 64,
                 temperature: float = 1.0, top_p: float = 0.9,
                 top_k: int = 0, eos_id: Optional[int] = None) -> torch.Tensor:
        self.eval()
        out = input_ids
        for _ in range(max_new_tokens):
            window = out[:, -self.cfg.model.max_seq_len:]
            logits, _ = self(window)
            nxt_logits = logits[:, -1, :].float() / max(1e-6, temperature)
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

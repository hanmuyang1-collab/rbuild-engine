"""R-Build v4: discrete-diffusion drafting and exact-AR verification.

The draft head is an absorbing-mask discrete diffusion transformer. It shares
both input embeddings and the LM output matrix with the main model. During
training, randomly masked targets provide a denoising objective and the main
model's detached logits provide a KL distillation objective.

Decoding is deliberately conservative: a proposal is only a performance hint.
The main model supplies the authoritative AR token at every position. Matching
critic-approved prefixes are committed early; the first mismatch (or exhausted
proposal) commits the authoritative token. Rejected tokens never reach the
watermark detector or the returned sequence.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.attention import RMSNorm


def _sinusoidal(values: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal encoding for positions or continuous diffusion times."""
    half = max(1, dim // 2)
    freq = torch.exp(torch.arange(half, device=values.device, dtype=torch.float32)
                     * (-math.log(10000.0) / half))
    angles = values.float().unsqueeze(-1) * freq
    out = torch.cat([angles.sin(), angles.cos()], dim=-1)
    return out[..., :dim]


class DraftBlock(nn.Module):
    """Bidirectional transformer block used only inside the draft horizon."""

    def __init__(self, dim: int, ffn_dim: int, n_heads: int, eps: float):
        super().__init__()
        self.norm1 = RMSNorm(dim, eps)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = RMSNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        return x + self.ffn(self.norm2(x))


class DiscreteDiffusionDraft(nn.Module):
    """Tiny absorbing-mask diffusion drafter with shared token embeddings."""

    def __init__(self, cfg, main_dim: int):
        super().__init__()
        self.cfg = cfg
        d = cfg.draft_verify
        self.main_dim = main_dim
        self.dim = cfg.resolved_draft_dim()
        self.ffn_dim = cfg.resolved_draft_ffn_dim()
        if self.dim % d.n_heads:
            raise ValueError("draft_verify.d_model must divide draft_verify.n_heads")
        self.input_proj = nn.Linear(main_dim, self.dim, bias=False)
        self.output_proj = nn.Linear(self.dim, main_dim, bias=False)
        self.mask_value = nn.Parameter(torch.zeros(1, 1, self.dim))
        self.time_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.blocks = nn.ModuleList([
            DraftBlock(self.dim, self.ffn_dim, d.n_heads, cfg.model.rmsnorm_eps)
            for _ in range(d.n_layers)])
        self.final_norm = RMSNorm(self.dim, cfg.model.rmsnorm_eps)
        nn.init.normal_(self.mask_value, std=0.02)

    def forward(self, context_ids: torch.Tensor, draft_ids: torch.Tensor,
                draft_mask: torch.Tensor, timestep: float,
                token_embedding: nn.Embedding,
                lm_weight: torch.Tensor) -> torch.Tensor:
        """Return draft-position logits (B, L, V); context positions are read-only."""
        context = self.input_proj(token_embedding(context_ids))
        proposed = self.input_proj(token_embedding(draft_ids))
        proposed = torch.where(draft_mask.unsqueeze(-1),
                               self.mask_value.to(proposed.dtype), proposed)
        x = torch.cat([context, proposed], dim=1)
        B, T, D = x.shape
        pos = torch.arange(T, device=x.device, dtype=torch.float32)
        x = x + _sinusoidal(pos, D).unsqueeze(0).to(x.dtype)
        t = torch.full((B,), float(timestep), device=x.device)
        x = x + self.time_proj(_sinusoidal(t, D).to(x.dtype)).unsqueeze(1)
        for block in self.blocks:
            x = block(x)
        h = self.output_proj(self.final_norm(x[:, -draft_ids.shape[1]:]))
        return F.linear(h, lm_weight)

    def training_loss(self, context_ids: torch.Tensor, targets: torch.Tensor,
                      teacher_logits: torch.Tensor,
                      token_embedding: nn.Embedding,
                      lm_weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Masked denoising CE plus main-model distillation KL on masked tokens."""
        valid = targets >= 0
        rate = float(torch.empty((), device=targets.device).uniform_(0.15, 1.0))
        mask = valid & (torch.rand(targets.shape, device=targets.device) < rate)
        # Every row with a valid target contributes at least one masked token.
        for b in range(targets.shape[0]):
            if valid[b].any() and not mask[b].any():
                noise = torch.rand(targets.shape[1], device=targets.device)
                noise = noise.masked_fill(~valid[b], -1.0)
                mask[b, int(noise.argmax())] = True
        if not mask.any():
            zero = teacher_logits.sum() * 0.0
            return zero, zero
        safe_targets = targets.clamp_min(0)
        logits = self(context_ids, safe_targets, mask, rate,
                      token_embedding, lm_weight)
        pred = logits[mask].float()
        truth = targets[mask]
        ce = F.cross_entropy(pred, truth)
        teacher = teacher_logits.detach()[mask].float()
        kl = F.kl_div(F.log_softmax(pred, dim=-1),
                      F.softmax(teacher, dim=-1), reduction="batchmean")
        return ce, kl

    @torch.no_grad()
    def propose(self, context_ids: torch.Tensor, length: int, steps: int,
                timestep_seed: int, token_embedding: nn.Embedding,
                lm_weight: torch.Tensor, sample_fn) -> torch.Tensor:
        """Iteratively unmask an L-token proposal in K diffusion steps."""
        B = context_ids.shape[0]
        device = context_ids.device
        max_context = max(1, self.cfg.model.max_seq_len - length)
        context_ids = context_ids[:, -max_context:]
        ids = torch.zeros(B, length, dtype=torch.long, device=device)
        mask = torch.ones(B, length, dtype=torch.bool, device=device)
        generator = torch.Generator(device=device)
        generator.manual_seed(int(timestep_seed) % (2**63 - 1))
        for step in range(steps):
            timestep = 1.0 - step / max(1, steps)
            logits = self(context_ids, ids, mask, timestep,
                          token_embedding, lm_weight)
            sampled = sample_fn(logits.reshape(-1, logits.shape[-1]),
                                generator=generator).view(B, length)
            confidence = F.softmax(logits.float(), dim=-1).max(dim=-1).values
            reveal = min(length, math.ceil(length * (step + 1) / steps))
            ranked = confidence.masked_fill(~mask, -1.0).argsort(dim=1, descending=True)
            choose = torch.zeros_like(mask)
            choose.scatter_(1, ranked[:, :reveal], True)
            choose &= mask
            ids = torch.where(choose, sampled, ids)
            mask = mask & ~choose
        # The final step normally clears every mask; this also covers K > L.
        if mask.any():
            logits = self(context_ids, ids, mask, 0.0,
                          token_embedding, lm_weight)
            sampled = sample_fn(logits.reshape(-1, logits.shape[-1]),
                                generator=generator).view(B, length)
            ids = torch.where(mask, sampled, ids)
        return ids


def v4_release_gate(acceptance_rate: float, ar_output, v4_output,
                    canary_before: float, canary_after: float,
                    min_acceptance: float = 0.60,
                    canary_tolerance: float = 0.0) -> Dict[str, object]:
    """Evaluate the three v4 release gates; higher canary scores are better."""
    def as_bytes(value):
        if isinstance(value, bytes):
            return value
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy().tobytes()
        return bytes(value)
    ar_bytes = as_bytes(ar_output)
    v4_bytes = as_bytes(v4_output)
    alpha_ok = float(acceptance_rate) >= float(min_acceptance)
    byte_exact = ar_bytes == v4_bytes
    canary_ok = float(canary_after) >= float(canary_before) - float(canary_tolerance)
    return {
        "passed": bool(alpha_ok and byte_exact and canary_ok),
        "acceptance_rate": float(acceptance_rate),
        "alpha_ok": bool(alpha_ok),
        "byte_exact": bool(byte_exact),
        "canary_ok": bool(canary_ok),
        "min_acceptance": float(min_acceptance),
    }


class DraftVerifyDecoder:
    """L=8/K=3/R_max=2 longest-prefix decoder with an exact AR fallback."""

    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int,
                 sampling: Dict[str, object], watermark: bool,
                 eos_id: Optional[int], images: Optional[torch.Tensor],
                 base_seed: int) -> torch.Tensor:
        model = self.model
        cfg = model.cfg.draft_verify
        if input_ids.shape[0] != 1:
            # Keep the public batch contract until a batched verifier lands.
            model.last_v4_stats = {"batch_fallback": True, "proposed": 0,
                                   "accepted": 0, "acceptance_rate": 0.0}
            return model._generate_ar(input_ids, max_new_tokens, sampling,
                                      watermark, eos_id, images, base_seed)
        out = input_ids
        produced = 0
        round_id = 0
        stats = {"rounds": 0, "proposed": 0, "accepted": 0,
                 "bonus_tokens": 0, "ar_fallback_tokens": 0,
                 "critic_fallback_tokens": 0, "target_forwards": 0,
                 "draft_forwards": 0, "batch_fallback": False}
        required_y = cfg.verify_y_critics or model.cfg.critic.y_critics

        def commit(token):
            nonlocal out, produced
            out = torch.cat([out, token], dim=1)
            produced += 1

        while produced < max_new_tokens:
            if round_id >= cfg.max_rounds:
                out = model._ar_one(out, sampling, watermark, images,
                                    base_seed, produced)
                produced += 1
                stats["ar_fallback_tokens"] += 1
                stats["target_forwards"] += 1
                round_id = 0
                if eos_id is not None and bool((out[:, -1] == eos_id).all()):
                    break
                continue

            length = min(cfg.draft_len, max_new_tokens - produced)
            draft_seed = (base_seed + 0xD134A7 * (stats["rounds"] + 1)) % (2**63 - 1)
            draft = model.draft.propose(
                out, length, cfg.denoise_steps, draft_seed,
                model.embed, model.lm_head.weight,
                lambda logits, generator: model._sample_logits(
                    model._prepare_target_logits(
                        logits, logits[:, 0], sampling, False), generator))
            stats["draft_forwards"] += cfg.denoise_steps
            stats["proposed"] += length

            window = torch.cat([out, draft], dim=1)[:, -model.cfg.model.max_seq_len:]
            hidden, _, _ = model._encode(window, images=images, stable_inference=True)
            logits = model._logits_from_hidden(hidden)
            _, votes = model.critics_verify(hidden)
            stats["target_forwards"] += 1
            prefix = window.shape[1] - length
            accepted_now = 0
            stopped = False

            for j in range(length):
                token_index = produced
                prev = window[:, prefix + j - 1]
                generator = model._token_rng(window.device, base_seed, token_index)
                target_logits = model._prepare_target_logits(
                    logits[:, prefix + j - 1], prev, sampling, watermark)
                target = model._sample_logits(target_logits, generator=generator)
                draft_token = draft[:, j:j + 1]
                critic_ok = votes[:, prefix + j] >= required_y
                if bool(((target == draft_token) & critic_ok).all()):
                    commit(draft_token)
                    accepted_now += 1
                else:
                    if bool((target == draft_token).all()):
                        stats["critic_fallback_tokens"] += 1
                    commit(target)
                    stopped = True
                if stopped or (eos_id is not None and bool((out[:, -1] == eos_id).all())):
                    stopped = True
                    break

            stats["accepted"] += accepted_now
            # A completely accepted proposal yields the usual speculative bonus.
            if (not stopped and accepted_now == length and produced < max_new_tokens):
                prev = window[:, -1]
                generator = model._token_rng(window.device, base_seed, produced)
                target_logits = model._prepare_target_logits(
                    logits[:, -1], prev, sampling, watermark)
                bonus = model._sample_logits(target_logits, generator=generator)
                commit(bonus)
                stats["bonus_tokens"] += 1

            stats["rounds"] += 1
            round_id += 1
            if eos_id is not None and bool((out[:, -1] == eos_id).all()):
                break

        stats["acceptance_rate"] = (stats["accepted"] / stats["proposed"]
                                    if stats["proposed"] else 0.0)
        stats["main_call_reduction_x"] = (produced / stats["target_forwards"]
                                          if stats["target_forwards"] else 1.0)
        model.last_v4_stats = stats
        return out

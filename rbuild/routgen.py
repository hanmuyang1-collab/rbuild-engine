"""
R-OutGen — the media-native generative model: the EXACT R-Build text
architecture, retargeted to produce media.

The trunk is not *similar* to R-Build's text model — it IS R-Build's text
model. ROutGenModel imports and builds the very same classes RBuildModel
uses:

    CacheLoopLine        — the extraction loop: critic-gated, ACT-halted,
                           pulling the fast-weight cache every loop
    ParallelBundleStage  — the generative stages: fine-grained MoE branches
                           (top-k routed experts + shared expert), MoD
                           routing, bundle gate, and the v3 stage critic
                           panels (more capacity than the working experts)
    FastWeightMemory     — the gradient-free delta-rule cache
    VisionTower          — optional prompt-side vision (ViT / encoderless /
                           VaWU), unchanged
    ThinkingModes        — modes still govern how hard the trunk thinks
                           before it renders

Same config object, same validation, same init scheme, same counter
invariant. Nothing about how the transformer thinks changes.

What changes is only what a media generator does not need — the text
OUTPUT side: the LM head, the actuation action head and the noting
experts are all text-vocabulary machinery, so R-OutGen does not build
them. In their place the v3.1 outgen heads are the model's NATIVE
output: each is a routed mixture of renderer experts (RendererMoE —
MoE applied to output modalities) that decodes the hidden states at a
head's placeholder run straight into pixels / frames / waveform:

    prompt tokens -> [exact R-Build trunk] -> hidden states
                  -> ImageOutHead / VideoOutHead / TTSOutHead -> media

Counter-verified: count_routgen_parameters() derives from
RBuildConfig.count_parameters() — the canonical counter — by removing
exactly the text-only pieces, so the printed number matches a built
ROutGenModel's sum(p.numel()) exactly.

Training works through the stock Trainer: forward accepts the same
4-tuple batches as the text model (out_targets is the objective; there
is no text CE), and checkpoints keep the same format
(model.pt + rbuild_config.json + meta.json).
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint

from .config import RBuildConfig
from .layers.attention import RMSNorm
from .memory import FastWeightMemory
from .model import CacheLoopLine, ParallelBundleStage     # the exact text trunk
from .outgen import OutGen
from .thinking import ThinkingModes
from .vision import VisionTower


# --------------------------------------------------------------------------- #
# counter — derived from the canonical text counter, project invariant
# --------------------------------------------------------------------------- #

def count_routgen_parameters(cfg: RBuildConfig) -> Dict[str, int]:
    """
    Parameter count of a ROutGenModel built from cfg. Derived from
    cfg.count_parameters() (the counter that mirrors RBuildModel exactly)
    by subtracting precisely the pieces R-OutGen does not build:

      - the noting experts        (text self-training machinery)
      - the actuation action head (text-vocab action columns)
      - the untied LM head        (only exists when tie_embeddings=False;
                                   tied embeddings are the shared embed table,
                                   which R-OutGen keeps for the prompt side)
    """
    c = cfg.count_parameters()
    drop = c["noting_params"] + c["actuation_params"]
    if not cfg.model.tie_embeddings:
        drop += cfg.effective_vocab_size() * cfg.model.d_model
    return {
        "total_params": int(c["total_params"] - drop),
        "active_params_per_token": int(c["active_params_per_token"] - drop),
        "trunk_params": int(c["cache_loop_params"] + c["parallel_params"]),
        "outgen_params": int(c["outgen_params"]),
        "embed_params": int(cfg.effective_vocab_size() * cfg.model.d_model),
        "vision_params": int(c["vision_params"]),
        "dropped_text_output_params": int(drop),
    }


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #

class ROutGenModel(nn.Module):
    """
    Media-native R-Build: the exact text architecture with renderer-MoE
    heads as its native output. Requires cfg.outgen.enabled=True with at
    least one of image/video/tts switched on.
    """

    def __init__(self, cfg: RBuildConfig):
        super().__init__()
        o = cfg.outgen
        assert o.enabled and (o.image or o.video or o.tts), \
            "ROutGenModel needs cfg.outgen.enabled=True and at least one " \
            "of outgen.image / outgen.video / outgen.tts"
        cfg.validate()
        self.cfg = cfg
        m = cfg.model
        eff_vocab = cfg.effective_vocab_size()

        # ---- the exact R-Build text trunk (same classes as RBuildModel) ----
        self.embed = nn.Embedding(eff_vocab, m.d_model)
        self.cache_loop = CacheLoopLine(cfg)
        self.stages = nn.ModuleList(ParallelBundleStage(cfg)
                                    for _ in range(cfg.parallel.n_stages))
        self.final_norm = RMSNorm(m.d_model, m.rmsnorm_eps)
        self.memory = FastWeightMemory(cfg.memory.key_dim, cfg.memory.value_dim,
                                       cfg.memory.write_lr, cfg.memory.decay,
                                       cfg.memory.max_facts) if cfg.memory.enabled else None
        self.fact_key_proj = nn.Linear(m.d_model, cfg.memory.key_dim, bias=False)
        self.fact_value_proj = nn.Linear(m.d_model, cfg.memory.value_dim, bias=False)
        self.drop = nn.Dropout(m.dropout)

        # prompt-side vision, unchanged from the text model
        self.vision = VisionTower(cfg) if cfg.vision.enabled else None
        if self.vision is not None and cfg.vision.freeze_vision:
            for name, p_ in self.vision.named_parameters():
                if "projector" not in name:
                    p_.requires_grad = False

        # ---- the NATIVE output: routed renderer-MoE heads (no lm_head) ----
        self.outgen = OutGen(cfg, m.d_model)

        # v3 runtime knobs driven by thinking modes (same as the text model)
        self._runtime_max_loops: Optional[int] = None
        self._runtime_y_critics: Optional[int] = None
        self._runtime_sampling: Optional[dict] = None
        self._active_thinking_mode: Optional[str] = None
        self.thinking_mode = ThinkingModes(self)
        if cfg.critic.enabled and cfg.thinking.default_mode:
            try:
                self.thinking_mode.apply(cfg.thinking.default_mode)
            except KeyError:
                pass

        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    # ------------------------------------------------------------------ #
    # encoder path — mirrors RBuildModel exactly
    # ------------------------------------------------------------------ #
    def _splice_vision(self, input_ids: torch.Tensor,
                       images: Optional[torch.Tensor]) -> torch.Tensor:
        """Same placeholder splicing as the text model (soft tokens in)."""
        emb = self.embed(input_ids.clamp_max(self.embed.num_embeddings - 1))
        if self.vision is None:
            return emb
        if images is None:
            mask = input_ids == self.cfg.vision.image_token_id
            if mask.any():
                raise ValueError("input contains image placeholders but no images were passed")
            return emb
        img_tokens = self.vision(images)                       # (B, K, D)
        B, K, D = img_tokens.shape
        mask = input_ids == self.cfg.vision.image_token_id     # (B, T)
        counts = mask.sum(1)
        if not (counts == K).all():
            raise ValueError(
                f"each sample needs exactly {K} image placeholders "
                f"(got {counts.tolist()}); one placeholder per vision soft token")
        emb = emb.clone()
        emb[mask] = img_tokens.reshape(-1, D).to(emb.dtype)
        return emb

    def _encode(self, input_ids: torch.Tensor,
                images: Optional[torch.Tensor] = None) -> torch.Tensor:
        """The shared trunk: embed -> extraction loop -> stages -> norm."""
        x = self._splice_vision(input_ids, images)
        x = self.drop(x)
        x, _ponder = self.cache_loop(x, self.memory,
                                     max_loops=self._runtime_max_loops,
                                     y_critics=self._runtime_y_critics)
        for stage in self.stages:
            if self.training and self.cfg.train.grad_checkpoint and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(stage, x, use_reentrant=False)
            else:
                x = stage(x)
        return self.final_norm(x)

    def hidden_states(self, input_ids: torch.Tensor,
                      images: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Public access to the trunk's final hidden states."""
        return self._encode(input_ids, images)

    # ------------------------------------------------------------------ #
    # forward — media is the objective (no text CE, there is no text head)
    # ------------------------------------------------------------------ #
    def forward(self, input_ids: torch.Tensor,
                targets: Optional[torch.Tensor] = None,
                images: Optional[torch.Tensor] = None,
                out_targets: Optional[Dict[str, torch.Tensor]] = None,
                decode: bool = False
                ) -> Tuple[Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]],
                           Optional[torch.Tensor]]:
        """
        (media_dict_or_None, loss_or_None). `targets` is accepted for
        Trainer.fit compatibility and ignored: R-OutGen has no text head,
        so the media MSE + renderer balance loss is the only objective.
        decode=True also returns each head's prediction over the samples
        carrying its placeholder run, as {key: (prediction, valid_mask)}.
        """
        if targets is not None and not out_targets:
            raise ValueError(
                "R-OutGen has no text head — pass out_targets= "
                "(the media MSE is the objective)")
        x = self._encode(input_ids, images)
        loss = None
        if out_targets:
            mse, bal = self.outgen.training_loss(input_ids, x, out_targets)
            loss = mse + bal
        outs = self.decode_all(input_ids, x) if decode else None
        return outs, loss

    def decode_all(self, input_ids: torch.Tensor, hidden: torch.Tensor
                   ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Every head's decode over the samples that carry its run."""
        outs = {}
        for key, head, tid in self.outgen._heads():
            mask = input_ids == tid
            valid = mask.sum(1) > 0
            if not valid.any():
                continue
            n_pos = int(mask.sum(1)[valid][0])
            h = hidden[valid][mask[valid]].view(-1, n_pos, hidden.shape[-1])
            pred, _ = head(h)
            outs[key] = (pred, valid)
        return outs

    # ------------------------------------------------------------------ #
    # generation: prompt + one placeholder run -> encode once -> decode
    # ------------------------------------------------------------------ #
    def _media_prompt(self, prompt_ids: torch.Tensor, token_id: int,
                      n_tokens: int) -> torch.Tensor:
        pad = torch.full((prompt_ids.shape[0], n_tokens), token_id,
                         dtype=torch.long, device=prompt_ids.device)
        ids = torch.cat([prompt_ids, pad], dim=1)
        assert ids.shape[1] <= self.cfg.model.max_seq_len, \
            f"prompt + {n_tokens} output placeholders exceeds max_seq_len"
        x = self.hidden_states(ids)
        return x[:, -n_tokens:]

    @torch.no_grad()
    def generate_image(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        """Prompt -> image tensor (B, 3, S, S) in [0, 1]."""
        assert self.outgen.image_head is not None, \
            "generate_image needs outgen.image=True"
        self.eval()
        h = self._media_prompt(prompt_ids, self.cfg.outgen.image_token_id,
                               self.cfg.n_image_out_tokens())
        return self.outgen.decode_image(h)

    @torch.no_grad()
    def generate_video(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        """Prompt -> clip tensor (B, F, 3, S, S) in [0, 1]."""
        assert self.outgen.video_head is not None, \
            "generate_video needs outgen.video=True"
        self.eval()
        h = self._media_prompt(prompt_ids, self.cfg.outgen.video_token_id,
                               self.cfg.n_video_out_tokens())
        return self.outgen.decode_video(h)

    @torch.no_grad()
    def generate_audio(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        """Prompt -> waveform (B, L) in [-1, 1] at cfg.outgen.sample_rate."""
        assert self.outgen.tts_head is not None, \
            "generate_audio needs outgen.tts=True"
        self.eval()
        h = self._media_prompt(prompt_ids, self.cfg.outgen.audio_token_id,
                               self.cfg.outgen.audio_tokens)
        return self.outgen.decode_audio(h)

    # ------------------------------------------------------------------ #
    # gradient-free fact writes — same contract as the text model
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def remember(self, token_ids: torch.Tensor, lr: Optional[float] = None) -> None:
        """Write tokenized content straight into the fast-weight cache."""
        assert self.memory is not None, "memory is disabled in this config"
        ids = token_ids.view(-1).to(next(self.parameters()).device)
        emb = self.embed(ids).mean(0, keepdim=True)
        self.memory.write(self.fact_key_proj(emb), self.fact_value_proj(emb), lr=lr)
        self.memory._fact_log.append({"n_tokens": int(ids.numel())})

    # ------------------------------------------------------------------ #
    # params + checkpoints (same format as the text model)
    # ------------------------------------------------------------------ #
    def num_params(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        return {"total_actual": total, **count_routgen_parameters(self.cfg)}

    def save_checkpoint(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))
        self.cfg.save(os.path.join(path, "rbuild_config.json"))
        meta = {"model_class": "ROutGenModel",
                "memory_writes": int(self.memory.n_writes)
                if self.memory is not None else 0}
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load_checkpoint(cls, path: str, device: Optional[str] = None) -> "ROutGenModel":
        cfg = RBuildConfig.load(os.path.join(path, "rbuild_config.json"))
        model = cls(cfg)
        state = torch.load(os.path.join(path, "model.pt"),
                           map_location=device or "cpu", weights_only=False)
        model.load_state_dict(state)
        return model

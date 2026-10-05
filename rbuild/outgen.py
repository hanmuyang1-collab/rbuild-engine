"""
R-Build outgen — generative OUTPUT heads: TTS (audio OUT), image OUT,
video OUT.

The input side of v3 made the LLM the vision encoder (encoderless); the
output side mirrors it — *decoderless output*. The transformer's own
hidden states at a head's placeholder positions (<image_out>,
<video_out>, <audio_out> token ids placed after the prompt) are decoded
straight into pixels / frames / waveform. No external codec, VAE, or
diffusion dependency.

Is MoE usable for these modalities? Yes — the same way it works for text
FFNs. Each head decodes through a *routed mixture of renderer experts*:
a learned router sends every output token to its top-k renderer experts
(color/texture, motion, prosody... specialize on their own), with a
switch-style load-balance aux loss. This is exactly the pattern modern
T2I/TTS/video backbones use; here it reuses R-Build's expert philosophy.

Training: model.forward(..., out_targets={"image"|"video"|"audio": T})
adds an MSE loss on the decoded output (normalized pixels [0,1],
waveform [-1,1]). Generation: model.generate_image(prompt_ids),
model.generate_video(prompt_ids), model.generate_audio(prompt_ids).

Everything is opt-in via cfg.outgen (enabled=False builds nothing) and
counter-verified by RBuildConfig.count_parameters().
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# routed mixture of renderer experts — one head's decoder
# --------------------------------------------------------------------------- #

class RendererMoE(nn.Module):
    """
    Top-k routed renderer experts. h (B, T, d_model) -> (B, T, out_dim).
    Returns (output, load_balance_loss).
    """

    def __init__(self, d_model: int, hidden: int, out_dim: int,
                 n_renderers: int, top_k: int):
        super().__init__()
        self.n_renderers, self.top_k = n_renderers, top_k
        self.router = nn.Linear(d_model, n_renderers, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(),
                          nn.Linear(hidden, out_dim))
            for _ in range(n_renderers)])

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        probs = F.softmax(self.router(h), dim=-1)              # (B, T, R)
        w, idx = probs.topk(self.top_k, dim=-1)                # (B, T, k)
        w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        out = torch.zeros(h.shape[0], h.shape[1],
                          self.experts[0][-1].out_features,
                          dtype=h.dtype, device=h.device)
        fracs = []
        for i, expert in enumerate(self.experts):
            sel = (idx == i)                                   # (B, T, k)
            coeff = (w * sel).sum(dim=-1, keepdim=True)        # (B, T, 1)
            fracs.append(sel.any(dim=-1).to(h.dtype).mean())
            out = out + coeff * expert(h)
        f = torch.stack(fracs)                                 # token fraction per expert
        P = probs.mean(dim=(0, 1))                             # mean router prob per expert
        balance = self.n_renderers * (f * P).sum()             # switch-style aux
        return out, balance


def unpatchify(patches: torch.Tensor, grid: int, patch: int) -> torch.Tensor:
    """(B, grid², patch²·3) -> (B, 3, S, S)."""
    B = patches.shape[0]
    S = grid * patch
    z = patches.view(B, grid, grid, patch, patch, 3)
    return z.permute(0, 5, 1, 3, 2, 4).reshape(B, 3, S, S)


# --------------------------------------------------------------------------- #
# the three heads
# --------------------------------------------------------------------------- #

class ImageOutHead(nn.Module):
    """Hidden states at <image_out> positions -> one image (B, 3, S, S)."""

    def __init__(self, cfg, d_model: int):
        super().__init__()
        o = cfg.outgen
        self.grid = cfg.out_image_grid()
        self.patch = o.image_patch
        self.n_tokens = self.grid * self.grid
        out_dim = self.patch * self.patch * 3
        self.pos = nn.Parameter(torch.zeros(1, self.n_tokens, d_model))
        self.renderer = RendererMoE(d_model, max(8, int(d_model * o.hidden_mult)),
                                    out_dim, o.n_renderers, o.top_k_renderers)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z, bal = self.renderer(h + self.pos.to(h.dtype))
        return unpatchify(z, self.grid, self.patch), bal


class VideoOutHead(nn.Module):
    """Hidden states at <video_out> positions -> a clip (B, F, 3, S, S)."""

    def __init__(self, cfg, d_model: int):
        super().__init__()
        o = cfg.outgen
        self.grid = cfg.out_image_grid()
        self.patch = o.image_patch
        self.frames = o.video_frames
        self.per_frame = self.grid * self.grid
        out_dim = self.patch * self.patch * 3
        self.frame_pos = nn.Parameter(torch.zeros(1, self.frames, d_model))
        self.patch_pos = nn.Parameter(torch.zeros(1, self.per_frame, d_model))
        self.renderer = RendererMoE(d_model, max(8, int(d_model * o.hidden_mult)),
                                    out_dim, o.n_renderers, o.top_k_renderers)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = h.shape[0]
        pos = (self.frame_pos.repeat_interleave(self.per_frame, dim=1)
               + self.patch_pos.repeat(1, self.frames, 1)).to(h.dtype)
        z, bal = self.renderer(h + pos)                        # (B, F·g², p²·3)
        z = z.view(B * self.frames, self.per_frame, -1)
        imgs = unpatchify(z, self.grid, self.patch)            # (B·F, 3, S, S)
        return imgs.view(B, self.frames, *imgs.shape[1:]), bal


class TTSOutHead(nn.Module):
    """Hidden states at <audio_out> positions -> waveform (B, L) in [-1, 1]."""

    def __init__(self, cfg, d_model: int):
        super().__init__()
        o = cfg.outgen
        self.n_tokens = o.audio_tokens
        self.chunk = o.audio_chunk
        self.pos = nn.Parameter(torch.zeros(1, self.n_tokens, d_model))
        self.renderer = RendererMoE(d_model, max(8, int(d_model * o.hidden_mult)),
                                    self.chunk, o.n_renderers, o.top_k_renderers)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z, bal = self.renderer(h + self.pos.to(h.dtype))       # (B, T, chunk)
        return torch.tanh(z.reshape(h.shape[0], -1)), bal


# --------------------------------------------------------------------------- #
# the module the model builds
# --------------------------------------------------------------------------- #

class OutGen(nn.Module):
    """Container for whichever output heads the config switched on."""

    def __init__(self, cfg, d_model: int):
        super().__init__()
        self.cfg = cfg
        o = cfg.outgen
        self.image_head = ImageOutHead(cfg, d_model) if o.image else None
        self.video_head = VideoOutHead(cfg, d_model) if o.video else None
        self.tts_head = TTSOutHead(cfg, d_model) if o.tts else None

    # ------------------------------------------------------------------ #
    def _heads(self):
        for key, head, tid in (("image", self.image_head, self.cfg.outgen.image_token_id),
                               ("video", self.video_head, self.cfg.outgen.video_token_id),
                               ("audio", self.tts_head, self.cfg.outgen.audio_token_id)):
            if head is not None:
                yield key, head, tid

    def placeholder_ids(self):
        return {tid for _, _, tid in self._heads()}

    # ------------------------------------------------------------------ #
    def training_loss(self, input_ids: torch.Tensor, hidden: torch.Tensor,
                      out_targets: Dict[str, torch.Tensor]
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MSE between each head's decode and its target, only over samples
        whose input actually carries that head's placeholder run.
        Returns (weighted_mse, weighted_balance).
        """
        o = self.cfg.outgen
        device = hidden.device
        mse_total = torch.zeros((), device=device)
        bal_total = torch.zeros((), device=device)
        for key, head, tid in self._heads():
            if key not in out_targets or out_targets[key] is None:
                continue
            mask = input_ids == tid                            # (B, T)
            counts = mask.sum(1)
            valid = counts > 0
            if not valid.any():
                continue
            n_pos = counts[valid][0].item()
            assert (counts[valid] == n_pos).all(), \
                f"outgen {key}: every sample needs the same {n_pos} placeholders"
            h = hidden[valid][mask[valid]].view(-1, n_pos, hidden.shape[-1])
            pred, bal = head(h)
            tgt = out_targets[key].to(device=device, dtype=pred.dtype)[valid]
            mse_total = mse_total + F.mse_loss(pred, tgt)
            bal_total = bal_total + bal
        return o.loss_weight * mse_total, o.moe_balance * bal_total

    # ------------------------------------------------------------------ #
    # decode helpers (generation): hidden at placeholder run -> media
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def decode_image(self, h: torch.Tensor) -> torch.Tensor:
        """(B, n_image_tokens, D) -> (B, 3, S, S) in [0, 1]."""
        img, _ = self.image_head(h)
        return img.clamp(0.0, 1.0)

    @torch.no_grad()
    def decode_video(self, h: torch.Tensor) -> torch.Tensor:
        """(B, n_video_tokens, D) -> (B, F, 3, S, S) in [0, 1]."""
        vid, _ = self.video_head(h)
        return vid.clamp(0.0, 1.0)

    @torch.no_grad()
    def decode_audio(self, h: torch.Tensor) -> torch.Tensor:
        """(B, audio_tokens, D) -> (B, L) in [-1, 1]."""
        wav, _ = self.tts_head(h)
        return wav.clamp(-1.0, 1.0)

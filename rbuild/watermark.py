"""
R-Build v3 — generation watermarking.

Prove a text came from *your* model. During sampling, each position's
previous token is hashed with a secret key to seed an RNG that splits the
vocabulary into a "green list" (fraction `gamma`) and the rest. Green-list
logits are boosted by `delta`, so the output quietly favors green tokens.
Anyone with the key can replay the split and run a one-proportion z-test:
watermarked text shows far more green tokens than chance.

    cfg.watermark.enabled = True
    cfg.watermark.key = "my-secret"
    model = RBuildModel(cfg)
    out = model.generate(ids)                       # watermarked

    from rbuild import WatermarkDetector
    det = WatermarkDetector(cfg.watermark, cfg.effective_vocab_size())
    det.detect(out)       # {'z_score': 7.3, 'p_value': 1e-13, 'watermarked': True}

No parameters are added — watermarking is pure sampling-time signal, so
the parameter counter and checkpoints are unaffected. `enabled=False`
(default) leaves generation bit-identical to unwatermarked sampling.
Per-call override: model.generate(ids, watermark=False).
"""

from __future__ import annotations

import hashlib
import math
from typing import Optional

import torch


def _seed_from(key: str, prev_token: int) -> int:
    """Deterministic 64-bit seed from the secret key + previous token."""
    digest = hashlib.sha256(f"{key}:{prev_token}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


class GreenListWatermark:
    """
    Sampling-time green-list watermark. One instance per model; applied
    inside RBuildModel.generate() when watermarking is active.
    """

    def __init__(self, cfg, vocab_size: int):
        self.key = cfg.key
        self.delta = cfg.delta
        self.gamma = cfg.gamma
        self.vocab_size = vocab_size

    # ------------------------------------------------------------------ #
    def green_mask(self, prev_token: int, device=None) -> torch.Tensor:
        """Bool (V,) mask of green-list tokens for the given previous token."""
        gen = torch.Generator().manual_seed(_seed_from(self.key, int(prev_token)))
        scores = torch.rand(self.vocab_size, generator=gen)
        mask = scores < self.gamma
        return mask.to(device) if device is not None else mask

    def bias_logits(self, logits: torch.Tensor,
                    prev_tokens: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, V) next-token logits; prev_tokens: (B,) last token ids.
        Returns logits with +delta on each sequence's green list.
        """
        out = logits.clone()
        for b in range(logits.shape[0]):
            mask = self.green_mask(int(prev_tokens[b]), logits.device)
            out[b] = out[b] + self.delta * mask.to(out.dtype)
        return out


class WatermarkDetector:
    """
    Replays the green-list split over a token sequence and runs the
    one-proportion z-test against the expected `gamma` fraction.
    """

    def __init__(self, cfg, vocab_size: int):
        self.key = cfg.key
        self.gamma = cfg.gamma
        self.z_threshold = cfg.z_threshold
        self.vocab_size = vocab_size

    def _is_green(self, prev_token: int, token: int) -> bool:
        gen = torch.Generator().manual_seed(_seed_from(self.key, int(prev_token)))
        scores = torch.rand(self.vocab_size, generator=gen)
        return bool(scores[int(token)] < self.gamma)

    def detect(self, token_ids, skip_prompt: int = 0) -> dict:
        """
        token_ids: 1-D ids (prompt + completion) or (1, T).
        skip_prompt: ignore the first N positions (pass your prompt length
        so prompt tokens don't dilute the statistic).
        Returns z_score, p_value, green_fraction, n_tokens, watermarked.
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.view(-1).tolist()
        ids = [int(t) for t in token_ids]
        ids = ids[skip_prompt:] if skip_prompt else ids
        n = max(0, len(ids) - 1)
        if n < 8:
            return {"z_score": 0.0, "p_value": 1.0, "green_fraction": 0.0,
                    "n_tokens": n, "watermarked": False,
                    "note": "too few tokens to detect (need >= 8)"}
        green = sum(1 for i in range(len(ids) - 1)
                    if self._is_green(ids[i], ids[i + 1]))
        expected = self.gamma * n
        var = n * self.gamma * (1 - self.gamma)
        z = (green - expected) / math.sqrt(var) if var > 0 else 0.0
        p = 0.5 * math.erfc(z / math.sqrt(2))          # one-sided p-value
        return {
            "z_score": round(z, 3),
            "p_value": p,
            "green_fraction": round(green / n, 4),
            "n_tokens": n,
            "watermarked": bool(z >= self.z_threshold),
        }

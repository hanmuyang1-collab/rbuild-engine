"""
R-Build v3 — native actuation: the model clicks by itself.

Tools are snails. Instead of asking an external harness to parse text and
move a mouse, v3 reserves a block of *action tokens* directly in the
(vocab-extended) output space. Generating one of these tokens *is* the
action — one forward step, one action, no parsing layer:

    token id                      action
    vocab_size + 0                wait
    vocab_size + 1 + g*g cells    click(x, y)   on a screen_grid x screen_grid grid
    ... + n_scroll                scroll(dy)    discrete scroll steps
    ... + 1                       type_begin    (following text tokens are typed)
    ... + 1                       type_end

Usage:
    codec = ActionCodec(vocab_size, screen_grid=64)
    cfg.model.vocab_size = codec.extended_vocab_size   # counter + head grow
    cfg.actuation.enabled = True
    model = RBuildModel(cfg)
    ...
    out = model.generate(ids, ...)
    for action in codec.decode_actions(out[0]):
        action.execute(pyautogui)   # or route to your own driver

Grounding comes from the vision side (encoderless VaWU recommended: the
screen is just frames, frames are just tokens), so the same forward that
sees the screen also clicks it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn


@dataclass
class Action:
    """A decoded native action."""
    kind: str                       # "wait" | "click" | "scroll" | "type_begin" | "type_end"
    x: Optional[float] = None       # normalized [0, 1]
    y: Optional[float] = None
    dy: Optional[int] = None        # scroll steps (signed)

    def execute(self, driver) -> None:
        """
        Run against any driver exposing pyautogui-like methods:
        moveTo/click/scroll (driver must map normalized coords to pixels).
        """
        if self.kind == "click":
            driver.moveTo(self.x, self.y)
            driver.click()
        elif self.kind == "scroll":
            driver.scroll(self.dy)
        elif self.kind == "wait":
            import time; time.sleep(0.1)
        # type_begin/type_end are stream markers handled by the caller

    def __repr__(self):
        if self.kind == "click":
            return f"Action(click, x={self.x:.3f}, y={self.y:.3f})"
        if self.kind == "scroll":
            return f"Action(scroll, dy={self.dy})"
        return f"Action({self.kind})"


class ActionCodec:
    """
    Maps between reserved token ids and native actions.

    Token layout above `vocab_size`:
        [0]               wait
        [1, 1+g*g)        click cell (i, j) on a g x g grid
        [1+g*g, +2*s)     scroll: s steps down, s steps up
        [last-1]          type_begin
        [last]            type_end
    """

    def __init__(self, vocab_size: int, screen_grid: int = 64,
                 scroll_steps: int = 8):
        self.vocab_size = vocab_size
        self.screen_grid = screen_grid
        self.scroll_steps = scroll_steps
        self.wait_id = vocab_size
        self.click_base = vocab_size + 1
        self.n_click = screen_grid * screen_grid
        self.scroll_base = self.click_base + self.n_click
        self.n_scroll = 2 * scroll_steps
        self.type_begin_id = self.scroll_base + self.n_scroll
        self.type_end_id = self.type_begin_id + 1
        self.extended_vocab_size = self.type_end_id + 1

    # ------------------------------------------------------------------ #
    def is_action(self, token_id: int) -> bool:
        return token_id >= self.vocab_size

    def encode_click(self, x: float, y: float) -> int:
        g = self.screen_grid
        i = min(g - 1, max(0, int(y * g)))
        j = min(g - 1, max(0, int(x * g)))
        return self.click_base + i * g + j

    def encode_scroll(self, dy: int) -> int:
        s = self.scroll_steps
        dy = max(-s, min(s, int(dy)))
        return self.scroll_base + (dy + s - 1) if dy != 0 else self.wait_id

    # ------------------------------------------------------------------ #
    def decode_actions(self, token_ids) -> List[Action]:
        """token_ids: 1-D iterable of ints; non-action tokens are skipped."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        actions: List[Action] = []
        g = self.screen_grid
        for t in token_ids:
            t = int(t)
            if t == self.wait_id:
                actions.append(Action("wait"))
            elif self.click_base <= t < self.scroll_base:
                cell = t - self.click_base
                i, j = divmod(cell, g)
                actions.append(Action("click", x=(j + 0.5) / g, y=(i + 0.5) / g))
            elif self.scroll_base <= t < self.type_begin_id:
                dy = (t - self.scroll_base) - self.scroll_steps + 1
                actions.append(Action("scroll", dy=dy))
            elif t == self.type_begin_id:
                actions.append(Action("type_begin"))
            elif t == self.type_end_id:
                actions.append(Action("type_end"))
        return actions


class ActuationHead(nn.Module):
    """
    Auxiliary head over the action block: at positions where the next token
    is an action token, this head's logits are trained (and can be sampled)
    natively. Kept separate from the tied LM head so the action space stays
    clean even when embeddings are tied.
    """

    def __init__(self, d_model: int, n_action_tokens: int):
        super().__init__()
        self.proj = nn.Linear(d_model, n_action_tokens, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> action logits (B, T, n_action_tokens)."""
        return self.proj(x)

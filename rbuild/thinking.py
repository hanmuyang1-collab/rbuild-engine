"""
R-Build v3 — the thinking-mode creator.

Thinking modes are named, user-creatable presets that retune the *adaptive
machinery* of v3 at runtime — no rebuild, no reload:

    model.thinking_mode.deep()                    # apply the built-in "deep"
    model.thinking_mode.deep(max_loops=12)        # apply with an override
    model.thinking_mode.deep(12)                  # positional value = max_loops
    model.thinking_mode.create("exam", max_loops=10, y_critics=3,
                               halt_threshold=0.9, temperature=0.2)
    model.thinking_mode.exam()                    # your mode is now native

A mode is just a dict of knobs:
    max_loops       — extraction-loop cap while the mode is active
    y_critics       — how many critics must be satisfied to halt
    halt_threshold  — critic score that counts as "satisfied"
    temperature     — default sampling temperature in this mode
    top_p           — default nucleus value
    self_observe    — whether noting/verification runs during generation
                      (explicitly applying a mode turns it on even when
                      noting.nsct=False; startup never does)
    effort          — v3.2: the mode's built-in REASONING EFFORT (float).
                      Invisible to end users; it states how much thinking
                      the mode was designed for. The effective effort is
                      this value x the selectable effort (see below).
    reasoning       — v3.2: chain-of-thought switch (bool). False =
                      instant, no thinking: the extraction loop runs the
                      minimum number of iterations and the sampling knobs
                      are taken as-is.

Selectable effort (v3.2)
------------------------
Separate from the mode's built-in (invisible) reasoning effort, the
*selectable* effort is a built-in ladder of named multipliers — by
default {"low": 0.5, "medium": 1.0, "high": 2.0} — that the caller picks
BY TAG, unlike the invisible reasoning effort:

    "summarize this {effort:'high'}"     # tag inside the prompt text
    model.generate(ids, effort="high")   # or as a per-call kwarg
    model.thinking_mode.set_effort("low")# or persistently

    effective effort = mode's built-in effort x selectable multiplier

And it is real, not cosmetic: the selectable multiplier scales the
extraction-loop cap, the number of critics that must agree, the halt
threshold, and the sampling sharpness — more effort literally means
deeper extraction and stricter verification (higher quality), less means
faster and shallower. At the default "medium" (x1.0) every mode behaves
exactly as its knobs declare.

Built-ins: instant (reasoning off), fast, balanced, deep, careful,
research. Create your own with create(); they persist on the model (and
into checkpoints via the config when saved with `thinking.save_modes`).
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable, Dict, Optional, Tuple


BUILTIN_MODES: Dict[str, Dict[str, Any]] = {
    # reasoning=False -> instant, no thinking (effort 0 by definition)
    "instant":  dict(max_loops=1,  y_critics=1, halt_threshold=0.5,
                     temperature=1.0, top_p=1.0, self_observe=False,
                     effort=0.0, reasoning=False),
    "fast":     dict(max_loops=2,  y_critics=1, halt_threshold=0.5,
                     temperature=1.0, top_p=0.9, self_observe=False,
                     effort=0.5, reasoning=True),
    "balanced": dict(max_loops=6,  y_critics=2, halt_threshold=0.6,
                     temperature=0.9, top_p=0.9, self_observe=True,
                     effort=1.0, reasoning=True),
    "deep":     dict(max_loops=12, y_critics=3, halt_threshold=0.75,
                     temperature=0.7, top_p=0.95, self_observe=True,
                     effort=2.0, reasoning=True),
    "careful":  dict(max_loops=16, y_critics=4, halt_threshold=0.85,
                     temperature=0.4, top_p=0.95, self_observe=True,
                     effort=2.5, reasoning=True),
    "research": dict(max_loops=24, y_critics=4, halt_threshold=0.9,
                     temperature=0.6, top_p=0.98, self_observe=True,
                     effort=4.0, reasoning=True),
}

# the built-in selectable-effort ladder (name -> multiplier); configs may
# extend it via thinking.selectable_efforts
DEFAULT_SELECTABLE_EFFORTS: Dict[str, float] = {"low": 0.5, "medium": 1.0,
                                                "high": 2.0}

# {effort:'high'} / {effort:"low"} / {effort: low} — the caller-facing tag
_EFFORT_TAG_RE = re.compile(r"\{effort\s*:\s*['\"]?([A-Za-z0-9_\-]+)['\"]?\s*\}")


def parse_effort_tag(text: str) -> Tuple[str, Optional[str]]:
    """
    Extract a selectable-effort tag from prompt text.
    Returns (clean_text, tag_or_None). The tag is just a name — validate it
    against the ladder with ThinkingModes.set_effort / generate(effort=...).
    """
    m = _EFFORT_TAG_RE.search(text)
    if not m:
        return text, None
    return _EFFORT_TAG_RE.sub("", text).strip(), m.group(1)


class _ModeSetter:
    """Callable handle returned for each mode: `thinking_mode.deep(12)`."""

    def __init__(self, registry: "ThinkingModes", name: str):
        self._registry = registry
        self._name = name

    def __call__(self, value: Optional[int] = None, **overrides):
        return self._registry.apply(self._name, max_loops_override=value,
                                    **overrides)

    def __repr__(self):
        return f"<thinking_mode {self._name!r}: {self._registry.modes[self._name]}>"


class ThinkingModes:
    """
    Registry + applier for thinking modes. Accessed as `model.thinking_mode`;
    `model.thinking_mode.<mode>(<value>)` applies a mode, and
    `model.thinking_mode.create(...)` mints new ones. Also owns the
    selectable-effort tag: `set_effort("high")` re-scales the active mode.
    """

    def __init__(self, model):
        object.__setattr__(self, "_model", model)
        modes = dict(BUILTIN_MODES)
        modes.update(model.cfg.thinking.custom_modes or {})
        object.__setattr__(self, "modes", modes)
        object.__setattr__(self, "active", None)
        # v3.2: the selectable-effort ladder (built in; config-extendable)
        ladder = dict(DEFAULT_SELECTABLE_EFFORTS)
        ladder.update(model.cfg.thinking.selectable_efforts or {})
        object.__setattr__(self, "selectable_efforts", ladder)
        tag = model.cfg.thinking.default_effort
        if tag not in ladder:
            tag = "medium" if "medium" in ladder else sorted(ladder)[0]
        object.__setattr__(self, "_effort_tag", tag)

    # ------------------------------------------------------------------ #
    def __getattr__(self, name: str) -> _ModeSetter:
        modes = object.__getattribute__(self, "modes")
        if name.startswith("_") or name not in modes:
            raise AttributeError(
                f"unknown thinking mode {name!r}; available: {sorted(modes)} "
                f"(or mint one with thinking_mode.create({name!r}, ...))")
        return _ModeSetter(self, name)

    # ------------------------------------------------------------------ #
    # selectable effort (tag-selected, multiplies the mode's effort)
    # ------------------------------------------------------------------ #
    def set_effort(self, tag: str) -> str:
        """
        Select an effort level by tag ("low" / "medium" / "high" / anything
        added to thinking.selectable_efforts). Re-applies the active mode so
        the new multiplier takes effect immediately.
        """
        if tag not in self.selectable_efforts:
            raise KeyError(
                f"unknown effort tag {tag!r}; available: "
                f"{sorted(self.selectable_efforts)} "
                f"(or add one via cfg.thinking.selectable_efforts)")
        object.__setattr__(self, "_effort_tag", tag)
        if self.active is not None:
            self.apply(self.active)
        return tag

    def current_effort(self) -> str:
        """The active selectable-effort tag."""
        return object.__getattribute__(self, "_effort_tag")

    def effective_effort(self, name: Optional[str] = None,
                         tag: Optional[str] = None) -> Optional[float]:
        """
        effective effort = mode's built-in (invisible) reasoning effort
        x the selectable multiplier. 0.0 when reasoning is off (instant).
        """
        name = name or self.active
        if name is None:
            return None
        mode = self.modes[name]
        if not mode.get("reasoning", True):
            return 0.0
        mult = self.selectable_efforts[tag or self.current_effort()]
        return float(mode.get("effort", 1.0)) * mult

    # ------------------------------------------------------------------ #
    def create(self, name: str, **knobs) -> Dict[str, Any]:
        """
        Mint a thinking mode. Any subset of:
        max_loops, y_critics, halt_threshold, temperature, top_p,
        self_observe, effort (float, built-in reasoning effort), reasoning
        (bool — False = instant, no thinking).
        Unknown knobs are rejected loudly; missing ones inherit 'balanced'.
        """
        valid = {"max_loops", "y_critics", "halt_threshold",
                 "temperature", "top_p", "self_observe",
                 "effort", "reasoning"}
        bad = set(knobs) - valid
        if bad:
            raise ValueError(f"unknown thinking-mode knobs {sorted(bad)}; "
                             f"valid: {sorted(valid)}")
        mode = dict(self.modes.get("balanced", {}))
        mode.update(knobs)
        if float(mode.get("effort", 1.0)) <= 0 and mode.get("reasoning", True):
            raise ValueError(f"mode {name!r} has effort<=0 but reasoning=True; "
                             f"give it effort>0 or set reasoning=False (instant)")
        self.modes[name] = mode
        # keep modes with the config so checkpoints carry them
        self._model.cfg.thinking.custom_modes[name] = mode
        return mode

    def apply(self, name: str, max_loops_override: Optional[int] = None,
              selectable: Optional[str] = None, **overrides) -> Dict[str, Any]:
        """Activate a mode on the model (runtime-only; no rebuild)."""
        if name not in self.modes:
            raise KeyError(f"unknown thinking mode {name!r}")
        mode = dict(self.modes[name])
        if max_loops_override is not None:
            mode["max_loops"] = int(max_loops_override)
        mode.update(overrides)
        # v3.2: modes minted before effort/reasoning existed get sane defaults
        mode.setdefault("effort", 1.0)
        mode.setdefault("reasoning", True)

        if selectable is not None:
            self.set_effort(selectable)     # validates + re-applies; returns below
        tag = self.current_effort()
        mult = self.selectable_efforts[tag]
        builtin_effort = float(mode["effort"])
        reasoning = bool(mode["reasoning"])

        model = self._model
        c = model.cfg.critic
        if not reasoning:
            # ---- instant, no thinking: minimal extraction, knobs as-is ----
            if builtin_effort > 0:
                raise ValueError(
                    f"mode {name!r} has reasoning=False but effort={builtin_effort}; "
                    f"instant modes carry effort=0.0")
            eff = 0.0
            max_loops = max(1, min(c.min_loops, int(mode["max_loops"])))
            y = 1
            threshold = float(mode["halt_threshold"])
            temperature = float(mode["temperature"])
            top_p = float(mode["top_p"])
        else:
            if builtin_effort <= 0:
                raise ValueError(
                    f"mode {name!r} has reasoning=True but effort={builtin_effort}; "
                    f"reasoning modes need effort>0")
            # effective effort = built-in x selectable. A mode's declared
            # knobs ARE its behavior at its own built-in effort, so the
            # selectable multiplier scales them relative to that design
            # point (medium = x1.0 -> exactly the declared knobs).
            eff = builtin_effort * mult
            s = mult
            max_mult = max(self.selectable_efforts.values())
            cap = max(c.max_loops,
                      int(round(int(mode["max_loops"]) * max_mult)))
            max_loops = int(min(cap, max(c.min_loops,
                                         round(int(mode["max_loops"]) * s))))
            y = int(min(c.n_critics,
                        max(1, round(int(mode["y_critics"]) * s))))
            # threshold saturates toward 1 as effort grows (stricter gate)
            threshold = float(min(0.99, 1.0 - (1.0 - float(mode["halt_threshold"])) ** s))
            # higher effort sharpens sampling (quality), lower relaxes it
            temperature = float(min(2.0, max(0.05,
                                             float(mode["temperature"]) / math.sqrt(s))))
            top_p = float(mode["top_p"])

        # retune the adaptive machinery live
        model._runtime_max_loops = max_loops
        if model.cache_loop.critics is not None:
            model.cache_loop.critics.threshold = threshold
        model._runtime_y_critics = y
        model._runtime_sampling = {"temperature": temperature, "top_p": top_p}
        model._runtime_self_observe = bool(mode["self_observe"])
        model._runtime_reasoning = reasoning
        model._runtime_effort = eff
        model._runtime_effort_tag = tag
        object.__setattr__(self, "active", name)
        model._active_thinking_mode = name
        return mode

    def current(self) -> Optional[str]:
        return object.__getattribute__(self, "active")

    def list(self) -> Dict[str, Dict[str, Any]]:
        return dict(self.modes)

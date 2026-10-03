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

Built-ins: fast, balanced, deep, careful, research. Create your own with
create(); they persist on the model (and into checkpoints via the config
when saved with `thinking.save_modes`).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional


BUILTIN_MODES: Dict[str, Dict[str, Any]] = {
    "fast":     dict(max_loops=2,  y_critics=1, halt_threshold=0.5,
                     temperature=1.0, top_p=0.9, self_observe=False),
    "balanced": dict(max_loops=6,  y_critics=2, halt_threshold=0.6,
                     temperature=0.9, top_p=0.9, self_observe=True),
    "deep":     dict(max_loops=12, y_critics=3, halt_threshold=0.75,
                     temperature=0.7, top_p=0.95, self_observe=True),
    "careful":  dict(max_loops=16, y_critics=4, halt_threshold=0.85,
                     temperature=0.4, top_p=0.95, self_observe=True),
    "research": dict(max_loops=24, y_critics=4, halt_threshold=0.9,
                     temperature=0.6, top_p=0.98, self_observe=True),
}


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
    `model.thinking_mode.create(...)` mints new ones.
    """

    def __init__(self, model):
        object.__setattr__(self, "_model", model)
        modes = dict(BUILTIN_MODES)
        modes.update(model.cfg.thinking.custom_modes or {})
        object.__setattr__(self, "modes", modes)
        object.__setattr__(self, "active", None)

    # ------------------------------------------------------------------ #
    def __getattr__(self, name: str) -> _ModeSetter:
        modes = object.__getattribute__(self, "modes")
        if name.startswith("_") or name not in modes:
            raise AttributeError(
                f"unknown thinking mode {name!r}; available: {sorted(modes)} "
                f"(or mint one with thinking_mode.create({name!r}, ...))")
        return _ModeSetter(self, name)

    # ------------------------------------------------------------------ #
    def create(self, name: str, **knobs) -> Dict[str, Any]:
        """
        Mint a thinking mode. Any subset of:
        max_loops, y_critics, halt_threshold, temperature, top_p, self_observe.
        Unknown knobs are rejected loudly; missing ones inherit 'balanced'.
        """
        valid = {"max_loops", "y_critics", "halt_threshold",
                 "temperature", "top_p", "self_observe"}
        bad = set(knobs) - valid
        if bad:
            raise ValueError(f"unknown thinking-mode knobs {sorted(bad)}; "
                             f"valid: {sorted(valid)}")
        mode = dict(self.modes.get("balanced", {}))
        mode.update(knobs)
        self.modes[name] = mode
        # keep modes with the config so checkpoints carry them
        self._model.cfg.thinking.custom_modes[name] = mode
        return mode

    def apply(self, name: str, max_loops_override: Optional[int] = None,
              **overrides) -> Dict[str, Any]:
        """Activate a mode on the model (runtime-only; no rebuild)."""
        if name not in self.modes:
            raise KeyError(f"unknown thinking mode {name!r}")
        mode = dict(self.modes[name])
        if max_loops_override is not None:
            mode["max_loops"] = int(max_loops_override)
        mode.update(overrides)

        model = self._model
        c = model.cfg.critic
        # retune the adaptive machinery live
        model._runtime_max_loops = mode["max_loops"]
        if model.cache_loop.critics is not None:
            model.cache_loop.critics.threshold = mode["halt_threshold"]
        model._runtime_y_critics = min(mode["y_critics"], c.n_critics)
        model._runtime_sampling = {"temperature": mode["temperature"],
                                   "top_p": mode["top_p"]}
        model._runtime_self_observe = bool(mode["self_observe"])
        object.__setattr__(self, "active", name)
        model._active_thinking_mode = name
        return mode

    def current(self) -> Optional[str]:
        return object.__getattribute__(self, "active")

    def list(self) -> Dict[str, Dict[str, Any]]:
        return dict(self.modes)

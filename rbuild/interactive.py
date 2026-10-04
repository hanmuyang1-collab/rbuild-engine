"""
R-Build interactive layer — the "completely user-interactive" surface.

In Colab / Jupyter:
    from rbuild import interactive
    ui = interactive.launch()          # full widget panel, every value editable

Without widgets (plain terminal / scripts):
    ui = interactive.launch()          # automatically falls back to prompts

Every field of every config section — including the v3 sections (critic,
noting, thinking, actuation, watermark) — is exposed. Changing a value
re-runs validation + the parameter counter + the naive-vs-optimized cost
report live. The panel can then build the model in one click.

Also included: `chat(...)` — an interactive generation session where the
user can tweak sampling values, switch thinking modes, write facts into
the fast-weight memory, and inspect self-learning stats live.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Optional

import torch

from .config import RBuildConfig, preset

_SECTIONS = ("model", "cache_loop", "parallel", "critic", "noting",
             "thinking", "actuation", "watermark", "memory", "vision", "train")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _fields(cfg: RBuildConfig):
    """Yield (section_name, section_obj, field) for every config value."""
    for sec_name in _SECTIONS:
        sec = getattr(cfg, sec_name)
        for f in dataclasses.fields(sec):
            yield sec_name, sec, f


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython  # noqa
        shell = get_ipython()
        return shell is not None and "IPKernelApp" in shell.config
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# widget panel
# --------------------------------------------------------------------------- #

class InteractivePanel:
    """Holds the config + widgets; `panel.model` after Build is clicked."""

    def __init__(self, cfg: Optional[RBuildConfig] = None):
        self.cfg = cfg or RBuildConfig()
        self.model = None
        self._widgets: Dict[str, Any] = {}

    # ---------------- widgets path ---------------- #
    def _launch_widgets(self):
        import ipywidgets as w
        from IPython.display import display

        cfg = self.cfg
        boxes = {}
        for sec_name, sec, f in _fields(cfg):
            val = getattr(sec, f.name)
            if isinstance(val, dict):
                continue   # structured fields (custom thinking modes) stay programmatic
            label = f.name
            if isinstance(val, bool):
                widget = w.Checkbox(value=val, description=label, indent=False)
            elif isinstance(val, int):
                widget = w.IntText(value=val, description=label,
                                   style={"description_width": "150px"})
            elif isinstance(val, float):
                widget = w.FloatText(value=val, description=label,
                                     style={"description_width": "150px"})
            else:  # strings / choices
                choices = {
                    "bundle_mode": ["gate", "mean", "concat"],
                    "branch_ffn": ["moe", "dense"],
                    "precision": ["fp32", "bf16", "fp8"],
                    "optimizer": ["muon", "adamw"],
                    "mode": ["vit", "encoderless"],
                    "default_mode": ["fast", "balanced", "deep", "careful", "research"],
                }.get(f.name)
                if choices:
                    widget = w.Dropdown(options=choices, value=val, description=label,
                                        style={"description_width": "150px"})
                else:
                    widget = w.Text(value="" if val is None else str(val),
                                    description=label, style={"description_width": "150px"})
            widget.observe(self._on_change(sec, f.name, widget), names="value")
            self._widgets[f"{sec_name}.{f.name}"] = widget
            boxes.setdefault(sec_name, []).append(widget)

        accordion = w.Accordion(children=[w.VBox(v) for v in boxes.values()])
        for i, name in enumerate(boxes):
            accordion.set_title(i, name)

        self._report = w.HTML()
        self._build_btn = w.Button(description="Build model", button_style="success")
        self._save_btn = w.Button(description="Save config")
        self._preset_dd = w.Dropdown(options=["tiny", "s1", "s2", "s3", "s4", "s5"],
                                     value="tiny", description="preset")
        self._build_btn.on_click(self._on_build)
        self._save_btn.on_click(lambda *_: (self.cfg.save("rbuild_config.json"),
                                            self._refresh(extra="saved -> rbuild_config.json")))
        self._preset_dd.observe(self._on_preset, names="value")

        display(w.VBox([
            w.HTML("<b>R-Build v3 — every value is yours to change</b>"),
            self._preset_dd,
            accordion,
            w.HBox([self._build_btn, self._save_btn]),
            self._report,
        ], layout=w.Layout(width="720px")))
        self._refresh()
        return self

    def _on_change(self, sec, name, widget):
        def handler(change):
            old = getattr(sec, name)
            new = change["new"]
            try:
                if old is None and isinstance(new, str) and new == "":
                    setattr(sec, name, None)
                elif isinstance(old, bool):
                    setattr(sec, name, bool(new))
                elif isinstance(old, int):
                    setattr(sec, name, int(new))
                elif isinstance(old, float):
                    setattr(sec, name, float(new))
                else:
                    setattr(sec, name, None if new == "" else new)
            except (ValueError, TypeError):
                return
            self._refresh()
        return handler

    def _on_preset(self, change):
        self.cfg = preset(change["new"])
        self._sync_widgets_from_cfg()
        self._refresh()

    def _sync_widgets_from_cfg(self):
        for sec_name, sec, f in _fields(self.cfg):
            key = f"{sec_name}.{f.name}"
            if key not in self._widgets:
                continue
            val = getattr(sec, f.name)
            self._widgets[key].value = val if val is not None else ""

    def _on_build(self, *_):
        from .model import RBuildModel
        try:
            self.cfg.validate()
            self.model = RBuildModel(self.cfg)
            n = sum(p.numel() for p in self.model.parameters())
            self._refresh(extra=f"model built: {n/1e6:.2f}M params")
        except Exception as e:  # surface config errors to the user, don't crash the cell
            self._refresh(extra=f"config error: {e}")

    def _refresh(self, extra: str = ""):
        try:
            txt = self.cfg.report().replace("\n", "<br>")
        except Exception as e:
            txt = f"<i>invalid config: {e}</i>"
        if extra:
            txt += f"<br><b>{extra}</b>"
        self._report.value = f"<pre style='font-size:12px'>{txt}</pre>"

    # ---------------- console fallback ---------------- #
    def _launch_console(self):
        cfg = self.cfg
        print("R-Build v3 — interactive configuration (console mode)")
        print("Press Enter to keep the [default]. Type 'done' at any prompt to finish.\n")
        for sec_name, sec, f in _fields(cfg):
            val = getattr(sec, f.name)
            if isinstance(val, dict):
                continue
            raw = input(f"{sec_name}.{f.name} [{val}]: ").strip()
            if raw.lower() == "done":
                break
            if raw == "":
                continue
            try:
                if isinstance(val, bool):
                    setattr(sec, f.name, raw.lower() in ("1", "true", "yes", "y", "on"))
                elif isinstance(val, int):
                    setattr(sec, f.name, int(raw))
                elif isinstance(val, float):
                    setattr(sec, f.name, float(raw))
                else:
                    setattr(sec, f.name, None if raw.lower() in ("none", "null") else raw)
            except ValueError:
                print(f"  ! invalid value for {f.name}, keeping {val}")
        print()
        try:
            print(cfg.report())
        except Exception as e:
            print(f"invalid config: {e}")
        return self


def launch(cfg: Optional[RBuildConfig] = None) -> InteractivePanel:
    """
    Open the interactive R-Build panel. Uses ipywidgets inside
    Colab/Jupyter; falls back to console prompts elsewhere.
    """
    panel = InteractivePanel(cfg)
    if _in_notebook():
        try:
            import ipywidgets  # noqa
            return panel._launch_widgets()
        except ImportError:
            print("ipywidgets not installed — `pip install ipywidgets` "
                  "for the widget panel; falling back to console mode.")
    return panel._launch_console()


# --------------------------------------------------------------------------- #
# interactive generation session
# --------------------------------------------------------------------------- #

def chat(model, encode, decode, max_new_tokens: int = 64) -> None:
    """
    Interactive generation loop with live-modifiable sampling + memory +
    thinking modes + self-learning stats.

    Commands (type at the prompt):
      /temp <f>      temperature            /topp <f>    nucleus sampling
      /topk <i>      top-k (0 = off)        /maxn <i>    max new tokens
      /mode <name>   thinking mode (fast/balanced/deep/careful/research,
                     or anything you minted with thinking_mode.create)
      /modes         list thinking modes
      /remember ...  write text into fast-weight memory (no training)
      /forget        reset the fast-weight memory
      /selfstats     critic-verified self-learning stats
      /nsct on|off           toggle non-separate continuous training
                             (self-training while running; default off)
      /watermark on|off      toggle generation watermarking
      /detect <text>         z-test text for your watermark
      /config        show the live config report
      /quit          exit
    Anything else is encoded, run through the extraction loop and
    generative stages, and decoded back.
    """
    temp, topp, topk, maxn = 1.0, 0.9, 0, max_new_tokens
    print("R-Build v3 interactive session — /help-style commands listed in docstring.")
    while True:
        try:
            user = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        if user.startswith("/"):
            parts = user.split(maxsplit=1)
            cmd, arg = parts[0], (parts[1] if len(parts) > 1 else "")
            if cmd == "/quit":
                break
            elif cmd == "/temp":
                temp = float(arg)
            elif cmd == "/topp":
                topp = float(arg)
            elif cmd == "/topk":
                topk = int(arg)
            elif cmd == "/maxn":
                maxn = int(arg)
            elif cmd == "/mode":
                getattr(model.thinking_mode, arg)()
                print(f"  [thinking] mode -> {arg}")
            elif cmd == "/modes":
                for name, knobs in model.thinking_mode.list().items():
                    print(f"  {name}: {knobs}")
            elif cmd == "/remember":
                model.remember(encode(arg))
                print("  [cache] fact written (gradient-free)")
            elif cmd == "/forget":
                model.memory.reset()
                print("  [cache] memory reset")
            elif cmd == "/selfstats":
                print(f"  [self-learn] {model.self_learn_stats()}")
            elif cmd in ("/nsct", "/autotrain"):
                model.set_nsct(arg.lower() in ("on", "1", "true", "y"))
                print(f"  [NSCT] {'on' if model.cfg.noting.nsct else 'off'}")
            elif cmd == "/watermark":
                model.cfg.watermark.enabled = arg.lower() in ("on", "1", "true", "y")
                print(f"  [watermark] {'on' if model.cfg.watermark.enabled else 'off'}")
            elif cmd == "/detect":
                from .watermark import WatermarkDetector
                det = WatermarkDetector(model.cfg.watermark,
                                        model.cfg.effective_vocab_size())
                print(f"  [watermark] {det.detect(encode(arg).view(-1))}")
            elif cmd == "/config":
                print(model.cfg.report())
            else:
                print(f"  unknown command {cmd}")
            continue
        ids = encode(user)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        out = model.generate(ids, max_new_tokens=maxn,
                             temperature=temp, top_p=topp, top_k=topk)
        print("rbuild>", decode(out[0, ids.shape[1]:]))


def main():
    print("R-Build v3 interactive CLI")
    panel = launch()
    if panel.model is not None:
        print("panel.model is ready.")


if __name__ == "__main__":
    main()

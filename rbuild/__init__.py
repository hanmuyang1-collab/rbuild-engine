"""
R-Build v2.0 — a fully user-modifiable, open-source LLM architecture and
training-speed engine.

Quick start
-----------
>>> from rbuild import RBuildConfig, RBuildModel, Trainer, preset
>>> cfg = preset("s1")                    # or RBuildConfig() and edit anything
>>> cfg.parallel.n_branches = 8           # every value is modifiable
>>> print(cfg.report())                   # params + naive-vs-optimized cost
>>> model = RBuildModel(cfg)

Interactive (Colab / Jupyter):
>>> from rbuild import interactive
>>> ui = interactive.launch()             # widget panel for every value
>>> model = ui.model                      # after clicking "Build model"

Architecture: a single-line cache-loop stage (loops to pull the fast-weight
delta-rule cache) feeds stacked parallel-bundle stages whose branches bundle
their outputs and push them onward.
"""

from .config import (RBuildConfig, ModelConfig, CacheLoopConfig,
                     ParallelConfig, MemoryConfig, TrainConfig, preset)
from .model import RBuildModel, CacheLoopLine, ParallelBundleStage
from .memory import FastWeightMemory
from .optim import Muon, WSDScheduler, build_optimizer
from .train import Trainer
from . import interactive

__version__ = "2.0.0"

__all__ = [
    "RBuildConfig", "ModelConfig", "CacheLoopConfig", "ParallelConfig",
    "MemoryConfig", "TrainConfig", "preset",
    "RBuildModel", "CacheLoopLine", "ParallelBundleStage",
    "FastWeightMemory", "Muon", "WSDScheduler", "build_optimizer",
    "Trainer", "interactive",
]

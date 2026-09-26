"""
R-Build v2.1 — a fully user-modifiable, open-source LLM architecture and
training-speed engine, now with complete VL support.

Quick start
-----------
>>> from rbuild import RBuildConfig, RBuildModel, Trainer, preset
>>> cfg = preset("s1")                    # or RBuildConfig() and edit anything
>>> cfg.parallel.n_branches = 8           # every value is modifiable
>>> cfg.vision.enabled = True             # v2.1: add the ViT tower (blind if False)
>>> print(cfg.report())                   # params + naive-vs-optimized cost
>>> model = RBuildModel(cfg)

Interactive (Colab / Jupyter):
>>> from rbuild import interactive
>>> ui = interactive.launch()             # widget panel for every value
>>> model = ui.model                      # after clicking "Build model"

Architecture: a single-line cache-loop stage (loops to pull the fast-weight
delta-rule cache) feeds stacked parallel-bundle stages whose branches bundle
their outputs and push them onward. Images/video enter as soft tokens spliced
at placeholder positions, so both stages stay modality-agnostic.
"""

from .config import (RBuildConfig, ModelConfig, CacheLoopConfig,
                     ParallelConfig, MemoryConfig, VisionConfig,
                     TrainConfig, preset)
from .model import RBuildModel, CacheLoopLine, ParallelBundleStage
from .memory import FastWeightMemory
from .vision import VisionTower
from .optim import Muon, WSDScheduler, build_optimizer
from .train import Trainer
from . import interactive

__version__ = "2.1.0"

__all__ = [
    "RBuildConfig", "ModelConfig", "CacheLoopConfig", "ParallelConfig",
    "MemoryConfig", "VisionConfig", "TrainConfig", "preset",
    "RBuildModel", "CacheLoopLine", "ParallelBundleStage",
    "FastWeightMemory", "VisionTower", "Muon", "WSDScheduler",
    "build_optimizer", "Trainer", "interactive",
]

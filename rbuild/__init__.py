"""
R-Build v3 — a fully user-modifiable, open-source LLM architecture and
training-speed engine: critic-gated adaptive extraction, critic-verified
non-separate self-training (opt-in), thinking modes, encoderless VL/VaWU,
native actuation, and generation watermarking.

v3.1 merges R-Run into this repo: the `rrun` package serves any R-Build
checkpoint with one command and hot-swaps the resident model with zero
server restart and a full KV cache (`rrun serve` / `rrun swap`).

Quick start
-----------
>>> from rbuild import RBuildConfig, RBuildModel, Trainer, preset
>>> cfg = preset("s1")                    # or RBuildConfig() and edit anything
>>> cfg.critic.n_critics = 8              # every value is modifiable
>>> cfg.vision.enabled = True             # add vision (vit or encoderless)
>>> print(cfg.report())                   # params + naive-vs-optimized cost
>>> model = RBuildModel(cfg)

v3 in one tour:
>>> model.thinking_mode.deep()            # ACT loops deepen, critics stricter
>>> model.thinking_mode.create("exam", max_loops=10, y_critics=3)
>>> model.thinking_mode.exam()            # your mode is now native
>>> out = model.generate(ids)             # noting experts observe, critics verify,
>>> model.self_learn_stats()              # verified notes already in memory
>>> trainer.self_train_step()             # consolidate notes into slow weights

Interactive (Colab / Jupyter):
>>> from rbuild import interactive
>>> ui = interactive.launch()             # widget panel for every value
>>> model = ui.model                      # after clicking "Build model"

Architecture: a single-line extraction loop (critic-gated, ACT-halted,
pulling the fast-weight delta-rule cache) feeds stacked parallel-bundle
generative stages, each watched by X critic experts with more capacity
than the working experts. Images/video enter as soft tokens (ViT tower or
encoderless, optional VaWU whole-video tokens); action tokens let the
model click natively.
"""

from .config import (RBuildConfig, ModelConfig, CacheLoopConfig,
                     ParallelConfig, CriticConfig, NotingConfig,
                     ThinkingConfig, ActuationConfig, WatermarkConfig,
                     MemoryConfig, VisionConfig, OutGenConfig,
                     TrainConfig, preset)
from .model import RBuildModel, CacheLoopLine, ParallelBundleStage
from .memory import FastWeightMemory
from .vision import VisionTower, VaWUPooler
from .outgen import OutGen, RendererMoE, ImageOutHead, VideoOutHead, TTSOutHead
from .routgen import ROutGenModel, count_routgen_parameters
from .critics import CriticExpert, CriticPanel, ACTHalting
from .noting import NotingExperts, VerifiedNoteBuffer, SelfLearner
from .thinking import ThinkingModes, parse_effort_tag
from .actuation import ActionCodec, ActuationHead, Action
from .watermark import GreenListWatermark, WatermarkDetector
from .optim import Muon, WSDScheduler, build_optimizer
from .data import (load_manifest, manifest_batches, hf_image_batches,
                   hf_video_batches, load_image, load_video, load_audio,
                   vision_tokens_per_sample)
from .train import Trainer
from . import interactive

__version__ = "3.2.0"

__all__ = [
    "RBuildConfig", "ModelConfig", "CacheLoopConfig", "ParallelConfig",
    "CriticConfig", "NotingConfig", "ThinkingConfig", "ActuationConfig",
    "WatermarkConfig", "MemoryConfig", "VisionConfig", "OutGenConfig",
    "TrainConfig", "preset",
    "RBuildModel", "CacheLoopLine", "ParallelBundleStage",
    "FastWeightMemory", "VisionTower", "VaWUPooler",
    "OutGen", "RendererMoE", "ImageOutHead", "VideoOutHead", "TTSOutHead",
    "ROutGenModel", "count_routgen_parameters",
    "CriticExpert", "CriticPanel", "ACTHalting",
    "NotingExperts", "VerifiedNoteBuffer", "SelfLearner",
    "ThinkingModes", "parse_effort_tag", "ActionCodec", "ActuationHead", "Action",
    "GreenListWatermark", "WatermarkDetector",
    "Muon", "WSDScheduler", "build_optimizer", "Trainer", "interactive",
    "load_manifest", "manifest_batches", "hf_image_batches",
    "hf_video_batches", "load_image", "load_video", "load_audio",
    "vision_tokens_per_sample",
]

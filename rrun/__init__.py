"""R-Run — one-command hosting & hot-swap serving engine for LLMs.

Part of R-Build v3.1 (rbuild-engine): R-Build builds and trains the models,
R-Run serves them — one repo, one engine. `rrun serve` hosts a checkpoint,
`rrun swap` hot-swaps the resident model with zero server restart and a
full KV cache for the new model.

Backends: rbuild (native R-Build checkpoints), vllm, hf (transformers),
mock (CPU smoke tests).
"""

__version__ = "0.2.0"

"""R-Build v2 layers: attention with MoD routing, fine-grained MoE, blocks."""

from .attention import CausalAttention, MoDRouter
from .moe import FineGrainedMoE, DenseFFN, build_ffn

__all__ = [
    "CausalAttention",
    "MoDRouter",
    "FineGrainedMoE",
    "DenseFFN",
    "build_ffn",
]

"""Residual-aware BiLLM-v2 quantization primitives."""

from .binarizer import binary_approximation
from .residual_selector import salient_scores, select_salient_mask
from .splitting import adaptive_split

__all__ = ["adaptive_split", "binary_approximation", "salient_scores", "select_salient_mask"]

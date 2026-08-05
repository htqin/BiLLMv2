"""End-to-end pure PTQ orchestration."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch

from billmv2.config import BiLLMv2Config

from .sequential import quantize_sequential


def set_reproducible(seed: int) -> None:
    """Seed supported random number generators deterministically."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_ptq_pipeline(
    model: torch.nn.Module,
    calibration_loader: list[Any],
    config: BiLLMv2Config,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Run pure no-gradient BiLLM-v2 PTQ."""

    set_reproducible(config.seed)
    if torch.is_grad_enabled():
        with torch.no_grad():
            return quantize_sequential(model, calibration_loader, config)
    return quantize_sequential(model, calibration_loader, config)

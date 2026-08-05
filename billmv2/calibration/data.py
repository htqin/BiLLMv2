"""Calibration pool loading through the existing BiLLM cache."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import torch

LOGGER = logging.getLogger(__name__)


def _import_billm_datautils() -> Any:
    root = Path(__file__).resolve().parents[2] / "external" / "BiLLM"
    if not root.exists():
        raise FileNotFoundError("external/BiLLM link is missing; run scripts/setup_links.sh")
    sys.path.insert(0, str(root))
    try:
        import datautils
    finally:
        sys.path.pop(0)
    return datautils


def get_candidate_pool(
    dataset: str,
    candidate_size: int,
    seed: int,
    seqlen: int,
    model: str,
) -> tuple[list[Any], Any]:
    """Load candidates without copying model or dataset data."""

    if candidate_size <= 0:
        raise ValueError("candidate_size must be positive")
    datautils = _import_billm_datautils()
    if dataset != "mixed":
        try:
            return datautils.get_loaders(
                dataset, nsamples=candidate_size, seed=seed, seqlen=seqlen, model=model
            )
        except (ConnectionError, FileNotFoundError, OSError):
            cache_root = Path(__file__).resolve().parents[2] / "external" / "BiLLM" / "cache"
            suffix = f"{model}.pt"
            cached_paths = [
                path for path in cache_root.rglob("*.pt")
                if dataset in str(path.parent) and str(path).endswith(suffix)
            ]
            if not cached_paths:
                raise
            LOGGER.warning("using existing token cache fallback: %s", cached_paths[0])
            train_loader, test_loader = torch.load(cached_paths[0])
            sliced = [
                (inputs[:, :seqlen], targets[:, :seqlen])
                for inputs, targets in train_loader[:candidate_size]
            ]
            return sliced, test_loader
    pools: list[Any] = []
    test_loader = None
    quota = (candidate_size + 1) // 2
    for offset, name in enumerate(("c4", "wikitext2")):
        try:
            samples, current_test = datautils.get_loaders(
                name, nsamples=quota, seed=seed + offset, seqlen=seqlen, model=model
            )
            pools.extend(samples)
            test_loader = test_loader or current_test
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
            LOGGER.warning("calibration source %s unavailable: %s", name, error)
    if not pools:
        raise RuntimeError("no mixed calibration source is available")
    return pools[:candidate_size], test_loader

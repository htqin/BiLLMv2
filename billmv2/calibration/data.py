"""Calibration pool loading through the existing BiLLM cache."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import torch

LOGGER = logging.getLogger(__name__)

CACHE_ROOT = Path("/autodl-fs/data/cclanro/billm-v2-output/cache")


def _cache_file(name: str, nsamples: int, seed: int, seqlen: int, model: str) -> Path:
    """Mirror the vendored datautils.get_loaders cache file naming."""

    return CACHE_ROOT / f"{name}_{nsamples}_{seed}_{seqlen}_{model}.pt"


def _import_billm_datautils() -> Any:
    root = Path(__file__).resolve().parents[2] / "external" / "BiLLM"
    if not root.exists():
        raise FileNotFoundError("vendored BiLLM sources are missing under external/BiLLM")
    sys.path.insert(0, str(root))
    try:
        import datautils

        original_get_loaders = datautils.get_loaders

        def get_loaders(name, nsamples=128, seed=0, seqlen=2048, model=""):
            cached = _cache_file(name, nsamples, seed, seqlen, model)
            if cached.exists():
                return torch.load(cached, weights_only=False)
            loaders = original_get_loaders(
                name, nsamples=nsamples, seed=seed, seqlen=seqlen, model=model
            )
            cached.parent.mkdir(parents=True, exist_ok=True)
            torch.save(loaders, cached)
            stale = Path("cache") / cached.name
            if stale.exists():
                stale.unlink()
            return loaders

        datautils.get_loaders = get_loaders
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
            suffix = f"{model}.pt"
            cached_paths = [
                path for path in CACHE_ROOT.rglob("*.pt")
                if dataset in str(path.parent) and str(path).endswith(suffix)
            ]
            if not cached_paths:
                raise
            LOGGER.warning("using existing token cache fallback: %s", cached_paths[0])
            train_loader, test_loader = torch.load(cached_paths[0], weights_only=False)
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

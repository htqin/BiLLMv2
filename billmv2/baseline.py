"""Compatibility boundary for model loading and BiLLM evaluation utilities."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch


def billm_root() -> Path:
    """Return the vendored, unmodified BiLLM compatibility source root."""

    root = Path(__file__).resolve().parents[1] / "external" / "BiLLM"
    if not root.exists():
        raise FileNotFoundError("vendored BiLLM sources are missing under external/BiLLM")
    return root


def load_model(model_name: str) -> torch.nn.Module:
    """Load an OPT or LLaMA causal LM using the existing Hugging Face cache."""

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
    model.seqlen = (
        2048
        if "llama" in model_name.lower()
        else int(getattr(model.config, "max_position_embeddings", 2048))
    )
    return model


def find_layers(module: torch.nn.Module) -> dict[str, torch.nn.Linear]:
    """Delegate layer discovery to the vendored BiLLM implementation."""

    root = billm_root()
    sys.path.insert(0, str(root))
    try:
        from modelutils import find_layers as original_find_layers
        return original_find_layers(module)
    finally:
        sys.path.pop(0)


def evaluate_perplexity(
    model: torch.nn.Module,
    test_loader: Any,
    device: str,
    dataset: str,
) -> float:
    """Evaluate perplexity with the original BiLLM evaluation implementation."""

    root = billm_root()
    sys.path.insert(0, str(root))
    try:
        from eval_ppl_utils import llama_eval, opt_eval
        if "opt" in model.config.model_type.lower():
            return float(opt_eval(model, test_loader, device, dataset, False))
        return float(llama_eval(model, test_loader, device, dataset, False))
    finally:
        sys.path.pop(0)


@torch.no_grad()
def evaluate_perplexity_limited(
    model: torch.nn.Module,
    test_loader: Any,
    device: str,
    max_samples: int,
) -> float:
    """Evaluate a bounded number of contiguous windows without copying BiLLM evaluation code."""

    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    input_ids = test_loader.input_ids
    seqlen = int(model.seqlen)
    available = input_ids.numel() // seqlen
    count = min(max_samples, available)
    if count == 0:
        raise ValueError("evaluation data is shorter than one sequence")
    model.to(device).eval()
    losses = []
    for index in range(count):
        tokens = input_ids[:, index * seqlen : (index + 1) * seqlen].to(device)
        logits = model(tokens, use_cache=False).logits
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.shape[-1]),
            tokens[:, 1:].reshape(-1),
            reduction="mean",
        )
        losses.append(loss)
    perplexity = float(torch.exp(torch.stack(losses).mean()))
    model.cpu()
    torch.cuda.empty_cache()
    if not torch.isfinite(torch.tensor(perplexity)):
        raise FloatingPointError("perplexity is NaN or Inf")
    return perplexity

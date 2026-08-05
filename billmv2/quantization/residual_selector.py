"""Residual-aware structured salient selection."""

from __future__ import annotations

import torch
from torch import Tensor


def salient_scores(
    weight: Tensor,
    residual: Tensor,
    hessian_diag: Tensor,
    metric: str,
    group_size: int = 1,
) -> Tensor:
    """Score contiguous input groups for structured salient selection."""

    if weight.shape != residual.shape or weight.ndim != 2:
        raise ValueError("weight and residual must be matching matrices")
    if hessian_diag.numel() != weight.shape[1] or group_size <= 0:
        raise ValueError("invalid Hessian diagonal or group size")
    if metric == "magnitude":
        element_score = weight.float().square()
    elif metric == "hessian":
        element_score = weight.float().square() * hessian_diag.float().unsqueeze(0)
    elif metric == "residual_hessian":
        element_score = residual.float().square() * hessian_diag.float().unsqueeze(0)
    else:
        raise ValueError(f"unsupported salient metric: {metric}")
    column_score = element_score.sum(dim=0)
    padding = (-column_score.numel()) % group_size
    if padding:
        column_score = torch.nn.functional.pad(column_score, (0, padding))
    return column_score.reshape(-1, group_size).sum(dim=1)


def select_salient_mask(
    scores: Tensor,
    shape: tuple[int, int],
    fraction: float,
    group_size: int = 1,
) -> Tensor:
    """Expand top-scoring input groups to a non-overlapping element mask."""

    if scores.ndim != 1 or not 0.0 <= fraction <= 1.0:
        raise ValueError("scores must be 1-D and fraction in [0, 1]")
    num_groups = scores.numel()
    count = min(num_groups, max(0, round(fraction * num_groups)))
    selected = torch.zeros(num_groups, dtype=torch.bool, device=scores.device)
    if count:
        indices = torch.argsort(scores, descending=True, stable=True)[:count]
        selected[indices] = True
    columns = selected.repeat_interleave(group_size)[: shape[1]]
    return columns.unsqueeze(0).expand(shape[0], -1).clone()

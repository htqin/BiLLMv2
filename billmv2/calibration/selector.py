"""Deterministic information-optimal calibration selectors."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SelectionResult:
    """Selected indices with marginal and cumulative information scores."""

    indices: list[int]
    marginal_scores: list[float]
    information_scores: list[float]


def _kcenter(features: Tensor, budget: int, seed: int) -> list[int]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    first = int(torch.randint(features.shape[0], (1,), generator=generator))
    indices = [first]
    distances = torch.cdist(features.float(), features[first : first + 1].float()).squeeze(1)
    while len(indices) < budget:
        candidate = int(torch.argmax(distances))
        indices.append(candidate)
        distances = torch.minimum(
            distances,
            torch.cdist(features.float(), features[candidate : candidate + 1].float()).squeeze(1),
        )
        distances[indices] = -1
    return indices


def _information_trace(features: Tensor, indices: list[int], eps: float) -> tuple[list[float], list[float]]:
    gram = torch.eye(features.shape[1], dtype=torch.float64) * eps
    previous = float(torch.linalg.slogdet(gram)[1])
    marginal: list[float] = []
    cumulative: list[float] = []
    for index in indices:
        vector = features[index].double()
        gram += torch.outer(vector, vector)
        score = float(torch.linalg.slogdet(gram)[1])
        marginal.append(score - previous)
        cumulative.append(score)
        previous = score
    return marginal, cumulative


def select_calibration(
    features: Tensor,
    budget: int,
    method: str = "hybrid",
    seed: int = 0,
    eps: float = 1e-5,
) -> SelectionResult:
    """Select a unique deterministic subset using random, k-center, or D-optimality."""

    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("features must be a non-empty matrix")
    if not 0 < budget <= features.shape[0] or eps <= 0:
        raise ValueError("invalid selection budget or epsilon")
    normalized = features.detach().float().cpu()
    generator = torch.Generator().manual_seed(seed)
    if method == "random":
        indices = torch.randperm(features.shape[0], generator=generator)[:budget].tolist()
    elif method == "kcenter":
        indices = _kcenter(normalized, budget, seed)
    elif method in {"d_optimal", "hybrid"}:
        inverse = torch.eye(features.shape[1], dtype=torch.float64) / eps
        selected = torch.zeros(features.shape[0], dtype=torch.bool)
        indices = []
        center_indices = _kcenter(normalized, budget, seed) if method == "hybrid" else []
        centered = normalized - normalized.mean(dim=0, keepdim=True)
        representation = 1.0 / (
            torch.cdist(centered, centered).mean(dim=1).clamp_min(eps)
        )
        representation /= representation.max().clamp_min(eps)
        for step in range(budget):
            vectors = normalized.double()
            gains = torch.log1p(torch.einsum("ni,ij,nj->n", vectors, inverse, vectors))
            if method == "hybrid":
                center_bonus = torch.zeros_like(gains)
                center_bonus[center_indices[step]] = gains.max().clamp_min(1.0) * 0.1
                gains = gains + 0.05 * representation.double() + center_bonus
            gains[selected] = -torch.inf
            index = int(torch.argmax(gains))
            indices.append(index)
            selected[index] = True
            vector = normalized[index].double()
            projected = inverse @ vector
            inverse -= torch.outer(projected, projected) / (1.0 + vector @ projected)
    else:
        raise ValueError(f"unsupported calibration selector: {method}")
    marginal, information = _information_trace(normalized, indices, eps)
    return SelectionResult(indices, marginal, information)

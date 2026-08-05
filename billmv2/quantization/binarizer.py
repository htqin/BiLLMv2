"""Hessian-aware structured binary approximations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class BinaryApproximation:
    """Store a two-term residual binary approximation."""

    quantized: Tensor
    signs: Tensor
    scales: Tensor


def _weighted_scale(values: Tensor, signs: Tensor, weights: Tensor, mask: Tensor) -> Tensor:
    numerator = (weights * mask * values * signs).sum(dim=1)
    denominator = (weights * mask * signs.square()).sum(dim=1)
    return torch.where(
        denominator > torch.finfo(torch.float32).eps,
        numerator / denominator,
        torch.zeros_like(numerator),
    )


def binary_approximation(
    weight: Tensor,
    mask: Tensor | None = None,
    hessian_diag: Tensor | None = None,
    order: int = 1,
) -> BinaryApproximation:
    """Fit one or two binary residual terms using weighted least squares."""

    if weight.ndim != 2 or order not in {1, 2}:
        raise ValueError("weight must be 2-D and order must be one or two")
    active = torch.ones_like(weight, dtype=torch.bool) if mask is None else mask.bool()
    if active.shape != weight.shape:
        raise ValueError("mask shape must match weight")
    column_weights = (
        torch.ones(weight.shape[1], device=weight.device, dtype=torch.float32)
        if hessian_diag is None
        else hessian_diag.float().to(weight.device)
    )
    if column_weights.numel() != weight.shape[1]:
        raise ValueError("Hessian diagonal must match input features")
    weights = column_weights.unsqueeze(0).expand_as(weight)
    terms: list[Tensor] = []
    scales: list[Tensor] = []
    residual = weight.float()
    for _ in range(order):
        signs = torch.where(residual >= 0, 1.0, -1.0) * active
        scale = _weighted_scale(residual, signs, weights, active)
        term = signs * scale.unsqueeze(1)
        terms.append(signs)
        scales.append(scale)
        residual = residual - term * active
    design = torch.stack(terms, dim=1)
    solved = torch.stack(scales, dim=1)
    if order == 2:
        # Joint 2x2 solve corrects correlation between the residual signs.
        a11 = (weights * design[:, 0].square()).sum(1)
        a22 = (weights * design[:, 1].square()).sum(1)
        a12 = (weights * design[:, 0] * design[:, 1]).sum(1)
        b1 = (weights * design[:, 0] * weight).sum(1)
        b2 = (weights * design[:, 1] * weight).sum(1)
        determinant = a11 * a22 - a12.square()
        stable = determinant.abs() > 1e-10 * (a11 * a22).clamp_min(1.0)
        safe_determinant = torch.where(stable, determinant, torch.ones_like(determinant))
        joint_first = (b1 * a22 - b2 * a12) / safe_determinant
        joint_second = (b2 * a11 - b1 * a12) / safe_determinant
        solved[:, 0] = torch.where(stable, joint_first, solved[:, 0])
        solved[:, 1] = torch.where(stable, joint_second, solved[:, 1])
    quantized = (design * solved.unsqueeze(-1)).sum(dim=1) * active
    return BinaryApproximation(quantized.to(weight), design.to(weight), solved.to(weight))

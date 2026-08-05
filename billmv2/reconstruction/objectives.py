"""Shared reconstruction metrics."""

from __future__ import annotations

import torch
from torch import Tensor


def geometry_weights(
    teacher_outputs: Tensor,
    mode: str,
    gamma: float = 0.5,
    eps: float = 1e-5,
) -> Tensor:
    """Build a diagonal soft-whitening metric from teacher outputs."""

    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if mode in {"none", "mse"}:
        return torch.ones(
            teacher_outputs.shape[-1],
            device=teacher_outputs.device,
            dtype=torch.float32,
        )
    if mode != "diagonal_whiten":
        raise ValueError(f"unsupported geometry loss: {mode}")
    flat = teacher_outputs.detach().float().reshape(-1, teacher_outputs.shape[-1])
    variance = flat.var(dim=0, unbiased=False)
    return (variance + eps).pow(-gamma / 2)


def geometry_loss(student_outputs: Tensor, teacher_outputs: Tensor, weights: Tensor) -> Tensor:
    """Compute a weighted output reconstruction loss."""

    if student_outputs.shape != teacher_outputs.shape:
        raise ValueError("teacher and student output shapes must match")
    if weights.ndim != 1 or weights.numel() != teacher_outputs.shape[-1]:
        raise ValueError("weights must match the output feature dimension")
    error = (teacher_outputs.float() - student_outputs.float()) * weights
    return error.square().mean()

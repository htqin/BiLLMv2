"""Activation-aware low-rank approximation of BiLLM residuals."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class LowRankFactors:
    """Store factors whose product is U @ V.T."""

    u: Tensor
    v: Tensor

    @property
    def rank(self) -> int:
        """Return factor rank."""

        return self.u.shape[1]

    def reconstruct(self) -> Tensor:
        """Reconstruct the low-rank matrix."""

        return self.u @ self.v.transpose(0, 1)


def weighted_low_rank(
    residual: Tensor,
    rank: int,
    metric: str = "diag_hessian",
    hessian: Tensor | None = None,
) -> LowRankFactors:
    """Fit a rank-constrained residual under weight or Hessian geometry."""

    if residual.ndim != 2 or not residual.is_floating_point():
        raise ValueError("residual must be a floating-point matrix")
    max_rank = min(residual.shape)
    if rank < 0 or rank > max_rank:
        raise ValueError(f"rank must be in [0, {max_rank}]")
    if rank == 0:
        return LowRankFactors(
            residual.new_zeros((residual.shape[0], 0)),
            residual.new_zeros((residual.shape[1], 0)),
        )
    work = residual.float()
    transform: Tensor | None = None
    inverse_transform: Tensor | None = None
    if metric == "diag_hessian":
        if hessian is None:
            raise ValueError("diag_hessian metric requires a Hessian")
        diagonal = torch.diag(hessian.float()) if hessian.ndim == 2 else hessian.float()
        if diagonal.numel() != residual.shape[1]:
            raise ValueError("Hessian diagonal must match input features")
        transform = diagonal.clamp_min(torch.finfo(torch.float32).eps).sqrt()
        work = work * transform.unsqueeze(0)
        inverse_transform = transform.reciprocal()
    elif metric == "full_hessian":
        if hessian is None or hessian.shape != (residual.shape[1], residual.shape[1]):
            raise ValueError("full_hessian requires a square input-feature Hessian")
        jitter = torch.finfo(torch.float32).eps * torch.eye(
            hessian.shape[0], device=hessian.device
        )
        transform = torch.linalg.cholesky(hessian.float() + jitter)
        work = work @ transform
        inverse_transform = torch.linalg.inv(transform)
    elif metric != "weight":
        raise ValueError(f"unsupported low-rank metric: {metric}")
    # A small deterministic randomized range finder avoids materializing a
    # full dense SVD for 7B-model projections.
    oversampled_rank = min(min(work.shape), rank + 4)
    generator = torch.Generator(device=work.device)
    generator.manual_seed(0)
    omega = torch.randn(
        work.shape[1], oversampled_rank, device=work.device,
        dtype=work.dtype, generator=generator,
    )
    basis, _ = torch.linalg.qr(work @ omega, mode="reduced")
    for _ in range(2):
        basis, _ = torch.linalg.qr(work @ (work.transpose(0, 1) @ basis), mode="reduced")
    small = basis.transpose(0, 1) @ work
    small_left, singular, right_h = torch.linalg.svd(small, full_matrices=False)
    left = basis @ small_left
    root = singular[:rank].sqrt()
    u = left[:, :rank] * root.unsqueeze(0)
    v_t = root.unsqueeze(1) * right_h[:rank]
    if metric == "diag_hessian":
        v_t = v_t * inverse_transform.unsqueeze(0)
    elif metric == "full_hessian":
        v_t = v_t @ inverse_transform
    return LowRankFactors(u.to(residual), v_t.transpose(0, 1).to(residual))

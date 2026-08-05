"""Finite-candidate symmetric and asymmetric binary splitting."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .binarizer import binary_approximation


@dataclass(frozen=True)
class SplitResult:
    """Store masks, reconstruction, and optional row-wise diagnostics."""

    concentrated_mask: Tensor
    sparse_mask: Tensor
    quantized: Tensor
    thresholds: Tensor | None
    error: float
    family: str = "asymmetric"
    candidate_histogram: list[int] | None = None
    row_diversity_ratio: float = 0.0
    mean_row_loss_reduction: float = 0.0
    boundary_candidate_ratio: float = 0.0


def _weighted_error(weight: Tensor, quantized: Tensor, diagonal: Tensor) -> Tensor:
    return ((weight.float() - quantized.float()).square() * diagonal.unsqueeze(0)).sum()


def adaptive_split(
    weight: Tensor,
    active_mask: Tensor,
    hessian_diag: Tensor,
    mode: str = "asymmetric",
    candidates: int = 16,
    rerank_topk: int = 4,
    inputs: Tensor | None = None,
    granularity: str = "global",
    row_rerank: str = "none",
    row_tile: int = 256,
) -> SplitResult:
    """Search a bounded threshold set and fit both binary branches."""

    if candidates < 1 or rerank_topk < 1 or row_tile < 1:
        raise ValueError("candidate counts and row tile must be positive")
    if granularity not in {"global", "per_row"}:
        raise ValueError("unsupported split granularity")
    if row_rerank not in {"none", "linear_top2"}:
        raise ValueError("unsupported row split rerank")
    active = active_mask.bool()
    if active.shape != weight.shape or hessian_diag.numel() != weight.shape[1]:
        raise ValueError("mask or Hessian shape mismatch")
    values = weight[active].float()
    if values.numel() == 0:
        empty = torch.zeros_like(weight)
        return SplitResult(active.clone(), active & False, empty, None, 0.0)
    if granularity == "per_row":
        return _adaptive_split_per_row(
            weight, active, hessian_diag, candidates, inputs, row_rerank, row_tile
        )
    quantiles = torch.linspace(0.1, 0.9, candidates, device=weight.device)
    families = (
        ("asymmetric", "mean_centered_asymmetric", "residual_balanced", "original_symmetric")
        if mode == "family_rerank" else ("original_symmetric",)
        if mode in {"original", "symmetric"} else (mode,)
    )
    coarse: list[tuple[Tensor, Tensor, Tensor, Tensor, str]] = []
    rerank_budget = 2 if mode == "family_rerank" else rerank_topk
    for family in families:
        source = weight.float()
        if family == "mean_centered_asymmetric":
            source = source - values.mean()
        elif family == "residual_balanced":
            base = binary_approximation(weight, active, hessian_diag, order=1).quantized
            source = (weight - base).float()
        elif family not in {"asymmetric", "original_symmetric"}:
            raise ValueError(f"unsupported split family: {family}")
        source_values = source[active]
        symmetric = torch.quantile(source_values.abs(), quantiles)
        if family == "original_symmetric":
            threshold_pairs = torch.stack((-symmetric, symmetric), dim=1)
        else:
            negative = source_values[source_values < 0].abs()
            positive = source_values[source_values >= 0]
            negative_q = torch.quantile(negative, quantiles) if negative.numel() else symmetric
            positive_q = torch.quantile(positive, quantiles) if positive.numel() else symmetric
            threshold_pairs = torch.stack((-negative_q, positive_q), dim=1)
        for pair in threshold_pairs:
            sparse = active & ((source < pair[0]) | (source > pair[1]))
            concentrated = active & ~sparse
            q_concentrated = binary_approximation(
                weight, concentrated, hessian_diag, order=1
            ).quantized
            q_sparse = binary_approximation(weight, sparse, hessian_diag, order=1).quantized
            quantized = q_concentrated + q_sparse
            coarse.append(
                (_weighted_error(weight * active, quantized, hessian_diag), pair,
                 concentrated, quantized, family)
            )
            if len(coarse) > rerank_budget:
                coarse.sort(key=lambda item: float(item[0]))
                del coarse[rerank_budget:]
    top = sorted(coarse, key=lambda item: float(item[0]))
    if inputs is not None:
        if inputs.shape[-1] != weight.shape[1]:
            raise ValueError("calibration inputs must match input features")
        def rerank(item: tuple[Tensor, Tensor, Tensor, Tensor, str]) -> float:
            error = inputs.float() @ (weight * active - item[3]).float().transpose(0, 1)
            return float(error.square().sum())
        best = min(top, key=rerank)
        final_error = rerank(best)
    else:
        best = top[0]
        final_error = float(best[0])
    _, pair, concentrated, quantized, family = best
    return SplitResult(
        concentrated, active & ~concentrated, quantized, pair, final_error, family
    )


def _masked_row_quantiles(values: Tensor, active: Tensor, quantiles: Tensor) -> tuple[Tensor, Tensor]:
    source = values.float()
    nan = torch.full_like(source, torch.nan)
    negative = torch.where(active & (source < 0), -source, nan)
    positive = torch.where(active & (source >= 0), source, nan)
    neg_q = torch.nanquantile(negative, quantiles, dim=1).transpose(0, 1)
    pos_q = torch.nanquantile(positive, quantiles, dim=1).transpose(0, 1)
    lower = torch.where(torch.isfinite(neg_q), -neg_q, torch.full_like(neg_q, -torch.inf))
    upper = torch.where(torch.isfinite(pos_q), pos_q, torch.full_like(pos_q, torch.inf))
    return lower, upper


def _row_loss(weight: Tensor, quantized: Tensor, active: Tensor, diagonal: Tensor) -> Tensor:
    target = weight.float() * active
    return ((target - quantized.float()).square() * diagonal.unsqueeze(0)).sum(dim=1)


def _adaptive_split_per_row(
    weight: Tensor,
    active: Tensor,
    hessian_diag: Tensor,
    candidates: int,
    inputs: Tensor | None,
    row_rerank: str,
    row_tile: int,
) -> SplitResult:
    rows, columns = weight.shape
    quantiles = torch.linspace(0.10, 0.90, candidates, device=weight.device)
    diagonal = hessian_diag.float().to(weight.device)
    best_loss = torch.full((rows,), torch.inf, device=weight.device)
    second_loss = torch.full((rows,), torch.inf, device=weight.device)
    best_candidate = torch.zeros((rows,), dtype=torch.long, device=weight.device)
    second_candidate = torch.zeros((rows,), dtype=torch.long, device=weight.device)
    best_sparse = torch.zeros_like(active)
    second_sparse = torch.zeros_like(active)
    best_quantized = torch.zeros_like(weight)
    second_quantized = torch.zeros_like(weight)
    global_candidate_loss = torch.zeros((candidates,), device=weight.device)

    for start in range(0, rows, row_tile):
        end = min(start + row_tile, rows)
        tile_weight = weight[start:end]
        tile_active = active[start:end]
        tile_rows = end - start
        lower_all, upper_all = _masked_row_quantiles(tile_weight, tile_active, quantiles)
        lower = lower_all.transpose(0, 1).unsqueeze(-1)
        upper = upper_all.transpose(0, 1).unsqueeze(-1)
        sparse_candidates = tile_active.unsqueeze(0) & (
            (tile_weight.float().unsqueeze(0) < lower)
            | (tile_weight.float().unsqueeze(0) > upper)
        )
        concentrated_candidates = tile_active.unsqueeze(0) & ~sparse_candidates
        expanded_weight = (
            tile_weight.unsqueeze(0)
            .expand(candidates, tile_rows, columns)
            .reshape(candidates * tile_rows, columns)
        )
        concentrated_flat = concentrated_candidates.reshape(candidates * tile_rows, columns)
        sparse_flat = sparse_candidates.reshape(candidates * tile_rows, columns)
        q_concentrated = binary_approximation(
            expanded_weight, concentrated_flat, diagonal, order=1
        ).quantized.reshape(candidates, tile_rows, columns)
        q_sparse = binary_approximation(
            expanded_weight, sparse_flat, diagonal, order=1
        ).quantized.reshape(candidates, tile_rows, columns)
        quantized_candidates = q_concentrated + q_sparse
        target = tile_weight.float().unsqueeze(0) * tile_active.unsqueeze(0)
        losses = (
            (target - quantized_candidates.float()).square()
            * diagonal.view(1, 1, columns)
        ).sum(dim=2)
        global_candidate_loss += losses.sum(dim=1)

        topk = min(2, candidates)
        top_values, top_indices = torch.topk(losses, k=topk, dim=0, largest=False)
        gather_best = top_indices[0].view(1, tile_rows, 1).expand(1, tile_rows, columns)
        best_loss[start:end] = top_values[0]
        best_candidate[start:end] = top_indices[0]
        best_sparse[start:end] = sparse_candidates.gather(0, gather_best).squeeze(0)
        best_quantized[start:end] = quantized_candidates.gather(0, gather_best).squeeze(0)
        if topk > 1:
            gather_second = top_indices[1].view(1, tile_rows, 1).expand(1, tile_rows, columns)
            second_loss[start:end] = top_values[1]
            second_candidate[start:end] = top_indices[1]
            second_sparse[start:end] = sparse_candidates.gather(0, gather_second).squeeze(0)
            second_quantized[start:end] = quantized_candidates.gather(0, gather_second).squeeze(0)
        else:
            second_loss[start:end] = top_values[0]
            second_candidate[start:end] = top_indices[0]
            second_sparse[start:end] = best_sparse[start:end]
            second_quantized[start:end] = best_quantized[start:end]

    final_sparse = best_sparse
    final_quantized = best_quantized
    final_loss = float(best_loss.sum())
    if row_rerank == "linear_top2" and inputs is not None and torch.isfinite(second_loss).any():
        if inputs.shape[-1] != columns:
            raise ValueError("calibration inputs must match input features")
        row_order = torch.argsort(best_loss, descending=True)
        rerank_candidates: list[tuple[float, Tensor, Tensor]] = []
        for fraction in (0.0, 0.02, 0.05, 0.10):
            count = int(round(rows * fraction))
            switched = row_order[:count]
            sparse = best_sparse.clone()
            quantized = best_quantized.clone()
            if count > 0:
                sparse[switched] = second_sparse[switched]
                quantized[switched] = second_quantized[switched]
            error = inputs.float() @ (weight.float() * active - quantized.float()).transpose(0, 1)
            rerank_candidates.append((float(error.square().sum()), sparse, quantized))
        final_loss, final_sparse, final_quantized = min(rerank_candidates, key=lambda item: item[0])

    active_rows = active.any(dim=1)
    histogram = torch.bincount(best_candidate.detach().cpu(), minlength=candidates).tolist()
    if active_rows.any():
        active_count = int(active_rows.sum())
        used = int(torch.unique(best_candidate[active_rows]).numel())
        diversity = float((best_candidate[active_rows] != best_candidate[active_rows][0]).float().mean())
        boundary = (best_candidate[active_rows] == 0) | (best_candidate[active_rows] == candidates - 1)
        boundary_ratio = float(boundary.float().mean())
    else:
        active_count = 0
        used = 0
        diversity = 0.0
        boundary_ratio = 0.0
    global_loss = float(global_candidate_loss.min())
    mean_gain = max(0.0, (global_loss - float(best_loss.sum())) / max(1, active_count))
    return SplitResult(
        active & ~final_sparse,
        final_sparse,
        final_quantized,
        None,
        final_loss,
        "row_wise_asymmetric",
        histogram,
        diversity if used > 1 else 0.0,
        mean_gain,
        boundary_ratio,
    )

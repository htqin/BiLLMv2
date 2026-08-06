"""Function-preserving head-wise rotations for LLaMA value/output pairs."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class VoRotation:
    """Store one orthogonal value-channel rotation per key/value head."""

    name: str
    matrices: Tensor


def _normalized_hadamard(size: int, device: torch.device) -> Tensor:
    if size <= 0 or size & (size - 1):
        raise ValueError("Hadamard size must be a positive power of two")
    matrix = torch.ones((1, 1), device=device, dtype=torch.float32)
    while matrix.shape[0] < size:
        matrix = torch.cat(
            (torch.cat((matrix, matrix), 1), torch.cat((matrix, -matrix), 1)), 0
        )
    return matrix / math.sqrt(size)


def _orthogonal_fallback(size: int, seed: int, device: torch.device) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    sample = torch.randn((size, size), generator=generator, dtype=torch.float64)
    matrix, _ = torch.linalg.qr(sample)
    return matrix.to(device=device, dtype=torch.float32)


def _signed_hadamard(size: int, seed: int, device: torch.device) -> Tensor:
    if size & (size - 1):
        return _orthogonal_fallback(size, seed, device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    signs = torch.randint(0, 2, (size,), generator=generator).float().mul_(2).sub_(1)
    permutation = torch.randperm(size, generator=generator)
    return _normalized_hadamard(size, device) @ torch.diag(signs.to(device))[:, permutation]


def make_vo_rotation_candidates(
    attention: nn.Module,
    context: Tensor,
    families: tuple[str, ...],
    seeds: tuple[int, ...],
) -> list[VoRotation]:
    """Create deterministic per-head V/O rotation candidates."""

    num_heads = int(getattr(attention, "num_heads", attention.config.num_attention_heads))
    num_kv_heads = int(
        getattr(attention, "num_key_value_heads", getattr(attention.config, "num_key_value_heads", num_heads))
    )
    head_dim = int(attention.head_dim)
    if context.shape[-1] != num_heads * head_dim:
        raise ValueError("attention context width does not match head geometry")
    device = context.device
    reshaped = context.detach().float().reshape(-1, num_heads, head_dim)
    groups = num_heads // num_kv_heads
    covariances = []
    for kv_head in range(num_kv_heads):
        values = reshaped[:, kv_head * groups : (kv_head + 1) * groups].reshape(-1, head_dim)
        values = values - values.mean(0, keepdim=True)
        covariances.append(values.transpose(0, 1) @ values / max(values.shape[0], 1))
    result: list[VoRotation] = []
    for family in families:
        family_seeds = (0,) if family == "identity" else seeds
        for seed in family_seeds:
            matrices = []
            for head, covariance in enumerate(covariances):
                current_seed = seed * 1009 + head
                if family == "identity":
                    matrix = torch.eye(head_dim, device=device)
                elif family == "signed_hadamard":
                    matrix = _signed_hadamard(head_dim, current_seed, device)
                elif family == "random_orthogonal":
                    matrix = _orthogonal_fallback(head_dim, current_seed, device)
                elif family == "covariance_hadamard":
                    _, eigenvectors = torch.linalg.eigh(covariance)
                    matrix = eigenvectors @ _signed_hadamard(head_dim, current_seed, device)
                else:
                    raise ValueError(f"unsupported V/O rotation family: {family}")
                matrices.append(matrix)
            suffix = "" if family == "identity" else f"_s{seed}"
            result.append(VoRotation(f"{family}{suffix}", torch.stack(matrices)))
    return result


def fold_vo_weights(
    v_weight: Tensor,
    o_weight: Tensor,
    rotation: VoRotation,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[Tensor, Tensor]:
    """Fold a head-wise rotation into V rows and O columns."""

    if v_weight.shape[0] != num_kv_heads * head_dim:
        raise ValueError("V projection output shape does not match KV heads")
    if o_weight.shape[1] != num_heads * head_dim:
        raise ValueError("O projection input shape does not match attention heads")
    if rotation.matrices.shape != (num_kv_heads, head_dim, head_dim):
        raise ValueError("rotation tensor has an invalid shape")
    v_heads = v_weight.float().reshape(num_kv_heads, head_dim, v_weight.shape[1])
    matrices = rotation.matrices.float()
    rotated_v = torch.einsum("hij,hik->hjk", matrices, v_heads)
    o_heads = o_weight.float().reshape(o_weight.shape[0], num_heads, head_dim)
    groups = num_heads // num_kv_heads
    expanded = matrices.repeat_interleave(groups, dim=0)
    rotated_o = torch.einsum("ohd,hde->ohe", o_heads, expanded)
    return rotated_v.reshape_as(v_weight).to(v_weight), rotated_o.reshape_as(o_weight).to(o_weight)

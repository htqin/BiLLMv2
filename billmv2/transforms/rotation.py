"""Memory-bounded orthogonal rotations."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def _hadamard(order: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    if order <= 0 or order & (order - 1):
        raise ValueError("Hadamard order must be a positive power of two")
    matrix = torch.ones((1, 1), device=device, dtype=dtype)
    while matrix.shape[0] < order:
        matrix = torch.cat(
            (torch.cat((matrix, matrix), 1), torch.cat((matrix, -matrix), 1)), 0
        )
    return matrix / math.sqrt(order)


def make_block_rotation(
    size: int,
    mode: str,
    block_size: int = 128,
    seed: int = 0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Construct a block-diagonal fixed orthogonal rotation."""

    if size <= 0 or block_size <= 0:
        raise ValueError("size and block_size must be positive")
    if mode == "none":
        return torch.eye(size, device=device, dtype=dtype)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = torch.zeros((size, size), device=device, dtype=dtype)
    for start in range(0, size, block_size):
        width = min(block_size, size - start)
        if mode == "hadamard" and width & (width - 1) == 0:
            block = _hadamard(width, torch.device(device), dtype)
        elif mode in {"hadamard", "random_orthogonal"}:
            sample = torch.randn((width, width), generator=generator, dtype=torch.float64)
            block, _ = torch.linalg.qr(sample)
            block = block.to(device=device, dtype=dtype)
        else:
            raise ValueError(f"unsupported fixed rotation: {mode}")
        result[start : start + width, start : start + width] = block
    return result


def fold_input_rotation(weight: Tensor, rotation: Tensor) -> Tensor:
    """Fold xR into a linear weight so xW equals (xR)(R^T W)."""

    if weight.ndim != 2 or rotation.shape != (weight.shape[1], weight.shape[1]):
        raise ValueError("rotation must match linear input features")
    return weight @ rotation


def apply_block_rotation(
    inputs: Tensor,
    mode: str,
    block_size: int = 128,
    seed: int = 0,
    inverse: bool = False,
) -> Tensor:
    """Apply a deterministic block rotation without materializing a dense matrix."""

    if mode == "none":
        return inputs
    generator = torch.Generator(device="cpu").manual_seed(seed)
    output = torch.empty_like(inputs)
    width_total = inputs.shape[-1]
    for start in range(0, width_total, block_size):
        width = min(block_size, width_total - start)
        if mode == "hadamard" and width & (width - 1) == 0:
            block = _hadamard(width, inputs.device, torch.float32)
        elif mode in {"hadamard", "random_orthogonal"}:
            sample = torch.randn((width, width), generator=generator, dtype=torch.float64)
            block, _ = torch.linalg.qr(sample)
            block = block.to(device=inputs.device, dtype=torch.float32)
        else:
            raise ValueError(f"unsupported fixed rotation: {mode}")
        if inverse:
            block = block.transpose(0, 1)
        output[..., start : start + width] = (
            inputs[..., start : start + width].float() @ block
        ).to(inputs.dtype)
    return output

"""Rank-budget allocation helpers."""

from __future__ import annotations

from collections.abc import Sequence


def allocate_uniform(total_rank: int, num_layers: int) -> Sequence[int]:
    """Allocate an integer rank budget deterministically across layers."""

    if total_rank < 0 or num_layers <= 0:
        raise ValueError("invalid rank budget or layer count")
    base, remainder = divmod(total_rank, num_layers)
    return [base + int(index < remainder) for index in range(num_layers)]

"""Pure selection helpers for functional low-rank screening."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable


@dataclass(frozen=True)
class AlternationDecision:
    """Record whether a second functional round is accepted."""

    accepted: bool
    branch_loss_reduction: float
    block_loss_reduction: float


def functional_fit_validation_split(
    nsamples: int, fit_samples: int, validation_samples: int
) -> tuple[list[int], list[int]]:
    """Return deterministic non-overlapping fit/validation indices."""

    if min(nsamples, fit_samples, validation_samples) < 0:
        raise ValueError("sample counts must be non-negative")
    if fit_samples + validation_samples > nsamples:
        raise ValueError("fit + validation exceeds selected samples")
    fit = list(range(fit_samples))
    validation = list(range(fit_samples, fit_samples + validation_samples))
    return fit, validation


def accept_functional_alternation(
    round0_branch_loss: float,
    round0_block_loss: float,
    round1_branch_loss: float,
    round1_block_loss: float,
) -> AlternationDecision:
    """Accept the second binary-LR round only on validation block improvement."""

    accepted = round1_block_loss < round0_block_loss
    return AlternationDecision(
        accepted=accepted,
        branch_loss_reduction=(round0_branch_loss - round1_branch_loss) if accepted else 0.0,
        block_loss_reduction=(round0_block_loss - round1_block_loss) if accepted else 0.0,
    )


@dataclass(frozen=True)
class TopupCandidate:
    """One adjacent global budget upgrade candidate."""

    block: int
    branch: str
    kind: str
    additional_bits: int
    validation_block_gain: float
    validation_branch_gain: float = 0.0
    additional_macs: int = 0

    @property
    def utility(self) -> float:
        if self.additional_bits <= 0:
            return float("-inf")
        return self.validation_block_gain / self.additional_bits


@dataclass(frozen=True)
class TopupResult:
    """Accepted global top-up upgrades and final accounting."""

    accepted: tuple[TopupCandidate, ...]
    parameter_bits_before: int
    parameter_bits_after: int
    remaining_bits: int


def global_budget_topup(
    parameter_bits: int,
    total_quantized_weights: int,
    target_parameter_bpw: float,
    candidates: Iterable[TopupCandidate],
    refresh: Callable[[TopupCandidate], Iterable[TopupCandidate]] | None = None,
) -> TopupResult:
    """Greedily accept positive block-gain upgrades without exceeding target BPW."""

    target_bits = int(target_parameter_bpw * total_quantized_weights)
    accepted: list[TopupCandidate] = []
    available = [c for c in candidates if c.validation_block_gain > 0 and c.additional_bits > 0]
    current_bits = parameter_bits
    while available:
        remaining = target_bits - current_bits
        affordable = [c for c in available if c.additional_bits <= remaining]
        if not affordable:
            break
        choice = max(affordable, key=lambda c: c.utility)
        accepted.append(choice)
        current_bits += choice.additional_bits
        available = [c for c in available if c.block != choice.block]
        if refresh is not None:
            available.extend(
                c for c in refresh(choice)
                if c.validation_block_gain > 0 and c.additional_bits > 0
            )
    return TopupResult(
        accepted=tuple(accepted),
        parameter_bits_before=parameter_bits,
        parameter_bits_after=current_bits,
        remaining_bits=target_bits - current_bits,
    )


def select_rotation_rescue_blocks(
    block_scores: Iterable[dict[str, float | int | bool]], max_blocks: int
) -> list[int]:
    """Select at most max_blocks blocks satisfying residual-aware rescue gates."""

    selected: list[tuple[float, int]] = []
    for item in block_scores:
        rank = int(item.get("rank", 0))
        block = int(item["block"])
        loss_percentile = float(item.get("loss_percentile", 0.0))
        high_tail = bool(item.get("high_tail_energy", False))
        low_rank_gain = float(item.get("rank4_to_rank8_gain", 1.0))
        gated = rank >= 8 or high_tail or loss_percentile >= 0.75 or low_rank_gain <= 0.0
        if gated:
            priority = loss_percentile + (0.25 if rank >= 8 else 0.0) + (0.25 if high_tail else 0.0)
            selected.append((priority, block))
    return [block for _, block in sorted(selected, reverse=True)[:max_blocks]]


def accept_rotation_candidate(
    identity_block_loss: float,
    candidate_block_loss: float,
    identity_lookahead_loss: float | None,
    candidate_lookahead_loss: float | None,
    margin: float,
) -> bool:
    """Accept non-identity rotation only with block margin and no lookahead regression."""

    if candidate_block_loss > identity_block_loss * (1.0 - margin):
        return False
    if identity_lookahead_loss is not None and candidate_lookahead_loss is not None:
        return candidate_lookahead_loss <= identity_lookahead_loss
    return True

import torch

from billmv2.low_rank.compensation import weighted_low_rank


def test_low_rank_reduces_weighted_error() -> None:
    torch.manual_seed(2)
    residual = torch.randn(12, 9)
    diagonal = torch.linspace(0.5, 2.0, 9)
    factors = weighted_low_rank(residual, 4, "diag_hessian", diagonal)
    error = ((residual - factors.reconstruct()).square() * diagonal).sum()
    baseline = (residual.square() * diagonal).sum()
    assert factors.rank == 4
    assert error <= baseline


def test_rank_zero_has_no_effect() -> None:
    residual = torch.randn(4, 5)
    factors = weighted_low_rank(residual, 0, "weight")
    assert factors.rank == 0
    assert torch.count_nonzero(factors.reconstruct()) == 0

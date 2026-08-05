import pytest
import torch

from billmv2.calibration.selector import select_calibration


@pytest.mark.parametrize("method", ["random", "kcenter", "d_optimal", "hybrid"])
def test_selection_is_unique_and_reproducible(method: str) -> None:
    torch.manual_seed(4)
    features = torch.randn(30, 8)
    first = select_calibration(features, 10, method, seed=9)
    second = select_calibration(features, 10, method, seed=9)
    assert first.indices == second.indices
    assert len(first.indices) == len(set(first.indices))
    assert all(
        right >= left
        for left, right in zip(first.information_scores, first.information_scores[1:])
    )

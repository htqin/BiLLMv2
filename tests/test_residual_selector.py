import torch

from billmv2.quantization.residual_selector import salient_scores, select_salient_mask


def test_residual_hessian_matches_manual_score() -> None:
    weight = torch.ones(2, 4)
    residual = torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 0.0, 1.0]])
    diagonal = torch.tensor([1.0, 2.0, 3.0, 4.0])
    score = salient_scores(weight, residual, diagonal, "residual_hessian", group_size=2)
    manual = (residual.square() * diagonal).sum(0).reshape(2, 2).sum(1)
    torch.testing.assert_close(score, manual)
    mask = select_salient_mask(score, tuple(weight.shape), 0.5, group_size=2)
    assert mask.shape == weight.shape
    assert mask.sum() == 4

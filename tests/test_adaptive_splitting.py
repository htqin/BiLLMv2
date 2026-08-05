import pytest
import torch

from billmv2.quantization.splitting import adaptive_split


@pytest.mark.parametrize(
    "weight,mask",
    [
        (torch.zeros(3, 7), torch.ones(3, 7, dtype=torch.bool)),
        (torch.rand(3, 7), torch.ones(3, 7, dtype=torch.bool)),
        (torch.randn(3, 7), torch.zeros(3, 7, dtype=torch.bool)),
    ],
)
def test_degenerate_splits_are_finite(weight: torch.Tensor, mask: torch.Tensor) -> None:
    result = adaptive_split(weight, mask, torch.ones(7), candidates=5)
    assert torch.isfinite(result.quantized).all()
    assert not torch.any(result.concentrated_mask & result.sparse_mask)
    assert torch.equal(result.concentrated_mask | result.sparse_mask, mask)


def test_reranked_result_is_a_candidate_minimum() -> None:
    torch.manual_seed(3)
    weight = torch.randn(4, 8)
    mask = torch.ones_like(weight, dtype=torch.bool)
    inputs = torch.randn(20, 8)
    result = adaptive_split(
        weight, mask, torch.ones(8), candidates=8, rerank_topk=8, inputs=inputs
    )
    actual = ((inputs @ (weight - result.quantized).t()).square()).sum()
    assert actual.item() == pytest.approx(result.error, rel=1e-5)



def _slow_row_split(weight: torch.Tensor, mask: torch.Tensor, diagonal: torch.Tensor, candidates: int):
    from billmv2.quantization.binarizer import binary_approximation

    quantiles = torch.linspace(0.10, 0.90, candidates, device=weight.device)
    sparse = torch.zeros_like(mask)
    quantized = torch.zeros_like(weight)
    losses = torch.full((weight.shape[0],), torch.inf, device=weight.device)
    ids = torch.zeros(weight.shape[0], dtype=torch.long, device=weight.device)
    for row in range(weight.shape[0]):
        active = mask[row]
        if not active.any():
            losses[row] = 0.0
            continue
        values = weight[row, active]
        negative = values[values < 0].abs()
        positive = values[values >= 0]
        for candidate, quantile in enumerate(quantiles):
            lower = -torch.quantile(negative, quantile) if negative.numel() else weight.new_tensor(-torch.inf)
            upper = torch.quantile(positive, quantile) if positive.numel() else weight.new_tensor(torch.inf)
            row_sparse = active & ((weight[row] < lower) | (weight[row] > upper))
            row_concentrated = active & ~row_sparse
            c = binary_approximation(weight[row:row+1], row_concentrated.unsqueeze(0), diagonal, order=1).quantized
            sp = binary_approximation(weight[row:row+1], row_sparse.unsqueeze(0), diagonal, order=1).quantized
            q = c + sp
            loss = (((weight[row:row+1] * active) - q).square() * diagonal.unsqueeze(0)).sum()
            if loss < losses[row]:
                losses[row] = loss
                sparse[row] = row_sparse
                quantized[row] = q.squeeze(0)
                ids[row] = candidate
    return sparse, quantized, losses, ids


def test_per_row_independence_and_loss_not_above_global() -> None:
    weight = torch.tensor([
        [-7.1759, -0.8646, -0.0080, 0.0838, 0.1495, 1.0441, 1.2379, 2.9323],
        [-2.1182, -0.6780, -0.1643, 0.0926, 0.1631, 0.2476, 1.1946, 2.2431],
    ])
    mask = torch.ones_like(weight, dtype=torch.bool)
    diagonal = torch.ones(weight.shape[1])
    row = adaptive_split(weight, mask, diagonal, candidates=8, granularity="per_row")
    global_result = adaptive_split(weight, mask, diagonal, candidates=8, granularity="global")
    _, _, _, ids = _slow_row_split(weight, mask, diagonal, 8)
    assert ids[0] != ids[1]
    assert row.error <= global_result.error + 1e-6


def test_per_row_bit_neutrality() -> None:
    weight = torch.randn(8, 16)
    mask = torch.ones_like(weight, dtype=torch.bool)
    diagonal = torch.ones(16)
    global_result = adaptive_split(weight, mask, diagonal, candidates=8, granularity="global")
    row_result = adaptive_split(weight, mask, diagonal, candidates=8, granularity="per_row")
    assert global_result.sparse_mask.numel() == row_result.sparse_mask.numel()
    assert global_result.concentrated_mask.shape == row_result.concentrated_mask.shape
    assert global_result.sparse_mask.shape == row_result.sparse_mask.shape


def test_per_row_artifact_independent_from_thresholds() -> None:
    from billmv2.utils.artifacts import reconstruct_weight
    from billmv2.utils.bits import pack_bits, pack_indices

    block = {
        "start": 0,
        "end": 4,
        "salient_indices": pack_indices(torch.tensor([0]), 4),
        "salient_signs": pack_bits(torch.ones(2, 4, dtype=torch.bool)),
        "concentrated_signs": pack_bits(torch.ones(1, 12, dtype=torch.bool)),
        "sparse_signs": pack_bits(torch.ones(1, 0, dtype=torch.bool)),
        "sparse_mask": pack_bits(torch.zeros(4, 4, dtype=torch.bool)),
        "salient_scales": torch.zeros(4, 2, dtype=torch.float16),
        "concentrated_scales": torch.ones(4, 1, dtype=torch.float16),
        "sparse_scales": torch.zeros(4, 1, dtype=torch.float16),
    }
    artifact = {
        "shape": [4, 4], "rotation": "none", "rotation_matrix": None,
        "rotation_block_size": 4, "rotation_seed": 0,
        "blocks": [block], "low_rank": {},
        "activation": {"bits": 16, "group_size": 4, "symmetric": True,
                       "clip_method": "mse", "feature_dim": 4},
    }
    first = reconstruct_weight(artifact)
    block["thresholds"] = torch.tensor([-1.0, 1.0], dtype=torch.float16)
    second = reconstruct_weight(artifact)
    assert torch.equal(first, second)


def test_per_row_degenerate_rows_are_finite() -> None:
    weight = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],
        [-1.0, -2.0, -3.0, -4.0],
        [0.0, 0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
    ])
    mask = torch.ones_like(weight, dtype=torch.bool)
    result = adaptive_split(weight, mask, torch.ones(4), candidates=6, granularity="per_row")
    assert torch.isfinite(result.quantized).all()
    assert torch.isfinite(torch.tensor(result.error))
    assert not torch.any(result.concentrated_mask & result.sparse_mask)
    assert torch.equal(result.concentrated_mask | result.sparse_mask, mask)


def test_vectorized_per_row_matches_slow_reference() -> None:
    torch.manual_seed(34)
    weight = torch.randn(5, 9)
    mask = torch.rand(5, 9) > 0.25
    diagonal = torch.linspace(0.5, 1.5, 9)
    fast = adaptive_split(weight, mask, diagonal, candidates=7, granularity="per_row", row_tile=2)
    sparse, quantized, losses, _ = _slow_row_split(weight, mask, diagonal, 7)
    assert torch.equal(fast.sparse_mask, sparse)
    assert torch.allclose(fast.quantized, quantized, atol=1e-6)
    assert fast.error == pytest.approx(float(losses.sum()), rel=1e-6, abs=1e-6)

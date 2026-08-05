import json
import tempfile
from pathlib import Path

import torch
from torch import nn

from billmv2.low_rank.functional import (
    decode_functional_factors,
    fit_functional_spectrum,
    functional_factor_payload,
    truncate_functional_spectrum,
)
from billmv2.low_rank.screening import (
    TopupCandidate,
    accept_functional_alternation,
    accept_rotation_candidate,
    functional_fit_validation_split,
    global_budget_topup,
    select_rotation_rescue_blocks,
)
from billmv2.utils.artifacts import (
    apply_billmv2_artifacts,
    load_billmv2_artifacts,
    save_run_artifacts,
)
from billmv2.utils.bits import artifact_bpw_breakdown, pack_bits, pack_indices


def test_nested_rank_truncation() -> None:
    torch.manual_seed(31)
    z = torch.randn(256, 24)
    true_u = torch.randn(18, 12)
    true_v = torch.randn(24, 12)
    residual = (z @ true_v) @ true_u.transpose(0, 1)
    spectrum = fit_functional_spectrum(z, residual, 12, ridge=1e-6, sketch_dim=24, max_tokens=256)
    f2 = truncate_functional_spectrum(spectrum, 2, z)
    f4 = truncate_functional_spectrum(spectrum, 4, z)
    f6 = truncate_functional_spectrum(spectrum, 6, z)
    assert f2.u.shape == (18, 2)
    assert f4.v.shape == (24, 4)
    assert torch.allclose(f2.u, f4.u[:, :2])
    assert torch.allclose(f4.v, f6.v[:, :4])
    losses = []
    for factors in (f2, f4, f6):
        pred = (z @ factors.v) @ factors.u.transpose(0, 1)
        losses.append(float((residual - pred).square().mean()))
    assert losses[2] <= losses[1] <= losses[0]


def test_per_rank_int8_scales() -> None:
    torch.manual_seed(32)
    u = torch.randn(16, 2) * torch.tensor([0.02, 5.0])
    v = torch.randn(12, 2) * torch.tensor([4.0, 0.03])
    factors = type("Factors", (), {"u": u, "v": v, "rank": 2})()
    tensor_payload, _, _ = functional_factor_payload(factors, "int8", "linear", "tensor")
    rank_payload, _, _ = functional_factor_payload(factors, "int8", "linear", "per_rank")
    tensor_u, tensor_v = decode_functional_factors(tensor_payload, torch.device("cpu"))
    rank_u, rank_v = decode_functional_factors(rank_payload, torch.device("cpu"))
    original = u @ v.transpose(0, 1)
    tensor_error = (original - tensor_u @ tensor_v.transpose(0, 1)).square().mean()
    rank_error = (original - rank_u @ rank_v.transpose(0, 1)).square().mean()
    assert rank_payload["u_scale"].shape == (2,)
    assert rank_payload["v_scale"].shape == (2,)
    assert rank_error <= tensor_error


def test_functional_binary_lr_alternation() -> None:
    accepted = accept_functional_alternation(10.0, 8.0, 9.5, 7.9)
    rejected = accept_functional_alternation(10.0, 8.0, 8.0, 8.1)
    assert accepted.accepted
    assert accepted.block_loss_reduction > 0
    assert not rejected.accepted
    assert rejected.branch_loss_reduction == 0.0


def test_global_budget_topup() -> None:
    candidates = [
        TopupCandidate(0, "o_proj", "rank", 10, 3.0),
        TopupCandidate(1, "down_proj", "salient", 30, 20.0),
        TopupCandidate(2, "o_proj", "rank", 5, -1.0),
    ]

    def refresh(choice: TopupCandidate):
        if choice.block == 0 and choice.kind == "rank":
            return [TopupCandidate(0, choice.branch, "combined", 5, 2.0)]
        return []

    result = global_budget_topup(100, 100, 1.5, candidates, refresh)
    assert result.parameter_bits_after <= 150
    assert all(candidate.validation_block_gain > 0 for candidate in result.accepted)
    assert [candidate.kind for candidate in result.accepted] == ["salient", "rank", "combined"]


def test_functional_fit_validation_split() -> None:
    fit, validation = functional_fit_validation_split(128, 96, 32)
    assert len(fit) == 96
    assert len(validation) == 32
    assert set(fit).isdisjoint(validation)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


def _tiny_artifacts(low_rank_payload: dict) -> dict:
    block = {
        "start": 0, "end": 4,
        "salient_indices": pack_indices(torch.tensor([0]), 4),
        "salient_signs": pack_bits(torch.ones(2, 4, dtype=torch.bool)),
        "concentrated_signs": pack_bits(torch.ones(1, 12, dtype=torch.bool)),
        "sparse_signs": pack_bits(torch.ones(1, 0, dtype=torch.bool)),
        "sparse_mask": pack_bits(torch.zeros(4, 4, dtype=torch.bool)),
        "salient_scales": torch.zeros(4, 2, dtype=torch.float16),
        "concentrated_scales": torch.zeros(4, 1, dtype=torch.float16),
        "sparse_scales": torch.zeros(4, 1, dtype=torch.float16),
        "thresholds": torch.zeros(2, dtype=torch.float16),
    }
    return {"linear": {
        "shape": [4, 4], "rotation": "none", "rotation_matrix": None,
        "rotation_block_size": 4, "rotation_seed": 0,
        "blocks": [block], "low_rank": low_rank_payload,
        "activation": {"bits": 16, "group_size": 4, "symmetric": True,
                       "clip_method": "mse", "feature_dim": 4,
                       "rotation": "none", "rotation_block_size": 4,
                       "rotation_seed": 0},
    }}


def test_new_artifact_backward_compatibility() -> None:
    torch.manual_seed(33)
    factors = type("Factors", (), {
        "u": torch.randn(4, 2), "v": torch.randn(4, 2), "rank": 2,
    })()
    payload, _, _ = functional_factor_payload(factors, "int8", "linear", "per_rank")
    artifacts = _tiny_artifacts(payload)
    inputs = torch.randn(3, 4)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        save_run_artifacts(path, {"model": "tiny"}, artifacts, {}, artifact_bpw_breakdown(artifacts), {})
        expected = _TinyModel()
        expected.linear.weight.data.zero_()
        apply_billmv2_artifacts(expected, load_billmv2_artifacts(path))
        expected_outputs = expected(inputs)
        config = json.loads((path / "config.json").read_text())
        config["artifact_format_version"] = 3
        (path / "config.json").write_text(json.dumps(config))
        restored = _TinyModel()
        restored.linear.weight.data.zero_()
        apply_billmv2_artifacts(restored, load_billmv2_artifacts(path))
        assert torch.equal(expected_outputs, restored(inputs))


def test_selective_rotation_gate() -> None:
    blocks = [
        {"block": 0, "rank": 4, "loss_percentile": 0.1, "rank4_to_rank8_gain": 0.5},
        {"block": 1, "rank": 8, "loss_percentile": 0.2, "rank4_to_rank8_gain": 0.5},
        {"block": 2, "rank": 4, "loss_percentile": 0.9, "rank4_to_rank8_gain": 0.5},
        {"block": 3, "rank": 4, "loss_percentile": 0.3, "high_tail_energy": True},
    ]
    selected = select_rotation_rescue_blocks(blocks, 2)
    assert len(selected) == 2
    assert 0 not in selected
    assert not accept_rotation_candidate(10.0, 9.95, 5.0, 5.0, 0.01)
    assert not accept_rotation_candidate(10.0, 9.8, 5.0, 5.1, 0.01)
    assert accept_rotation_candidate(10.0, 9.8, 5.0, 5.0, 0.01)

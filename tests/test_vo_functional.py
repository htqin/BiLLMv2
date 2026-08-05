import math
import tempfile
from pathlib import Path

import torch
from torch import nn

from billmv2.config import BiLLMv2Config
from billmv2.low_rank.functional import (
    fit_functional_low_rank,
    functional_factor_payload,
    install_functional_adapter,
)
from billmv2.pipeline.sequential import _run_layer
from billmv2.quantization.joint_search import (
    capture_teacher_branches,
    local_candidate_parameter_bits,
)
from billmv2.transforms.vo_rotation import VoRotation, fold_vo_weights
from billmv2.utils.artifacts import (
    apply_billmv2_artifacts,
    load_billmv2_artifacts,
    save_run_artifacts,
)
from billmv2.utils.bits import artifact_bpw_breakdown, pack_bits, pack_indices


def _vo_error(dtype: torch.dtype) -> float:
    torch.manual_seed(4)
    device = torch.device("cuda" if dtype != torch.float32 and torch.cuda.is_available() else "cpu")
    heads, head_dim, hidden = 4, 8, 32
    v_weight = torch.randn(heads * head_dim, hidden, dtype=dtype, device=device)
    o_weight = torch.randn(hidden, heads * head_dim, dtype=dtype, device=device)
    matrices = []
    for head in range(heads):
        matrix = torch.roll(torch.eye(head_dim), shifts=head + 1, dims=1)
        matrix[:, ::2] *= -1
        matrices.append(matrix.to(device))
    rotation = VoRotation("test", torch.stack(matrices).to(dtype))
    rotated_v, rotated_o = fold_vo_weights(
        v_weight, o_weight, rotation, heads, heads, head_dim
    )
    inputs = torch.randn(2, 5, hidden, dtype=dtype, device=device)
    attention = torch.softmax(torch.randn(2, heads, 5, 5, device=device).float(), -1).to(dtype)
    x, a = inputs.float(), attention.float()
    values = (x @ v_weight.float().transpose(0, 1)).reshape(2, 5, heads, head_dim)
    context = torch.einsum("bhqk,bkhd->bqhd", a, values)
    expected = context.reshape(2, 5, -1) @ o_weight.float().transpose(0, 1)
    values_rotated = (x @ rotated_v.float().transpose(0, 1)).reshape(2, 5, heads, head_dim)
    context_rotated = torch.einsum("bhqk,bkhd->bqhd", a, values_rotated)
    actual = context_rotated.reshape(2, 5, -1) @ rotated_o.float().transpose(0, 1)
    return float((expected.float() - actual.float()).norm() / expected.float().norm())


def test_vo_rotation_invariance() -> None:
    assert _vo_error(torch.float32) < 1e-5
    assert _vo_error(torch.float16) < 2e-3
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        assert _vo_error(torch.bfloat16) < 2e-3


def test_rotation_fold_no_runtime_op() -> None:
    linear = nn.Linear(16, 16, bias=False)
    assert not any("rotation" in type(module).__name__.lower() for module in linear.modules())


def test_functional_low_rank_solver() -> None:
    torch.manual_seed(8)
    z = torch.randn(1024, 48)
    true_u = torch.randn(32, 4)
    true_v = torch.randn(48, 4)
    residual = (z @ true_v) @ true_u.transpose(0, 1)
    factors = fit_functional_low_rank(z, residual, 4, 1e-6, 48, 1024, 3)
    prediction = (z @ factors.v) @ factors.u.transpose(0, 1)
    assert factors.rank == 4
    assert (residual - prediction).square().mean() < residual.square().mean() * 1e-3


class _TupleAttention(nn.Module):
    def forward(self, inputs: torch.Tensor, **_: object):
        return inputs * 2, None, None


class _Mlp(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * 3


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _TupleAttention()
        self.mlp = _Mlp()

    def forward(self, inputs: torch.Tensor, **_: object):
        hidden = inputs + self.self_attn(inputs)[0]
        return (hidden + self.mlp(hidden),)


def test_branch_target_capture() -> None:
    block = _Block()
    inputs = torch.randn(3, 4, 5)
    outputs, targets = capture_teacher_branches(block, inputs, {}, _run_layer, 3)
    assert torch.allclose(targets.attention.float(), inputs * 2, atol=2e-3, rtol=2e-3)
    assert torch.allclose(targets.mlp.float(), (inputs + inputs * 2) * 3, atol=2e-3, rtol=2e-3)
    assert not torch.equal(targets.attention.float(), outputs.float())
    assert not torch.equal(targets.mlp.float(), outputs.float())


def test_local_fixed_bpw() -> None:
    o = nn.Linear(4096, 4096, bias=False)
    down = nn.Linear(11008, 4096, bias=False)
    baseline = local_candidate_parameter_bits(o, 128, 13, 0, "int8")
    baseline += local_candidate_parameter_bits(down, 128, 13, 0, "int8")
    candidate = local_candidate_parameter_bits(o, 128, 9, 4, "int8")
    candidate += local_candidate_parameter_bits(down, 128, 9, 4, "int8")
    assert candidate <= baseline


def test_functional_adapter_no_dense_merge() -> None:
    torch.manual_seed(12)
    linear = nn.Linear(16, 12, bias=False)
    base = linear.weight.detach().clone()
    factors = fit_functional_low_rank(
        torch.randn(64, 16), torch.randn(64, 12), 2, sketch_dim=8
    )
    payload, u, v = functional_factor_payload(factors, "int8", "linear")
    install_functional_adapter(linear, payload)
    inputs = torch.randn(4, 16)
    expected = inputs @ base.transpose(0, 1) + (inputs @ v) @ u.transpose(0, 1)
    assert torch.allclose(linear(inputs), expected, atol=2e-4, rtol=2e-4)
    assert torch.equal(linear.weight, base)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


def test_low_rank_artifact_reload() -> None:
    torch.manual_seed(15)
    factors = fit_functional_low_rank(
        torch.randn(32, 4), torch.randn(32, 4), 2, sketch_dim=4
    )
    payload, _, _ = functional_factor_payload(factors, "int8", "linear")
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
    artifacts = {"linear": {
        "shape": [4, 4], "rotation": "none", "rotation_matrix": None,
        "rotation_block_size": 4, "rotation_seed": 0,
        "blocks": [block], "low_rank": payload,
        "activation": {"bits": 16, "group_size": 4, "symmetric": True,
                       "clip_method": "mse", "feature_dim": 4,
                       "rotation": "none", "rotation_block_size": 4,
                       "rotation_seed": 0},
    }}
    model = _TinyModel()
    model.linear.weight.data.zero_()
    install_functional_adapter(model.linear, payload)
    inputs = torch.randn(3, 4)
    expected = model(inputs)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        save_run_artifacts(
            path, {"model": "tiny"}, artifacts, {},
            artifact_bpw_breakdown(artifacts), {},
        )
        restored = _TinyModel()
        apply_billmv2_artifacts(restored, load_billmv2_artifacts(path))
        actual = restored(inputs)
    assert torch.equal(expected, actual)

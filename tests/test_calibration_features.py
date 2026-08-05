from types import SimpleNamespace

import torch
from torch import nn

from billmv2.calibration.features import (
    extract_calibration_features,
    pooled_probe_feature,
)


class _TinyBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(width, width, bias=False)
        self.mlp = nn.Module()
        self.mlp.down_proj = nn.Linear(width, width, bias=False)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor]:
        hidden = self.self_attn.q_proj(hidden)
        return (self.mlp.down_proj(hidden),)


class _TinyProbeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="llama", hidden_size=8)
        self.embed = nn.Embedding(32, 8)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_TinyBlock(8), _TinyBlock(8)])

    def forward(self, tokens: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        hidden = self.embed(tokens)
        for layer in self.model.layers:
            hidden = layer(hidden)[0]
        return hidden


def test_binary_error_zero_and_changes_with_provisional() -> None:
    torch.manual_seed(1)
    hidden = torch.randn(2, 4, 8)
    weight = torch.randn(6, 8)
    activation_projection = torch.randn(8, 3)
    error_projection = torch.randn(6, 3)
    zero = pooled_probe_feature(
        hidden,
        weight,
        weight.clone(),
        activation_projection,
        error_projection,
        "binary_error",
    )
    changed = pooled_probe_feature(
        hidden,
        weight,
        torch.zeros_like(weight),
        activation_projection,
        error_projection,
        "binary_error",
    )
    assert zero.abs().max() < 1e-7
    assert not torch.allclose(zero, changed)


def test_features_reproducible_distinct_and_hooks_removed() -> None:
    torch.manual_seed(2)
    model = _TinyProbeModel()
    samples = [torch.tensor([[1, 2, 3, 4]]), torch.tensor([[5, 6, 7, 8]])]
    before = sum(len(module._forward_pre_hooks) for module in model.modules())
    first = extract_calibration_features(
        model, samples, "cpu", feature_dim=4, probe_stride=1, seed=9
    )
    second = extract_calibration_features(
        model, samples, "cpu", feature_dim=4, probe_stride=1, seed=9
    )
    after = sum(len(module._forward_pre_hooks) for module in model.modules())
    torch.testing.assert_close(first, second)
    assert not torch.allclose(first[0], first[1])
    assert before == after == 0


def test_activation_error_feature_tracks_a4() -> None:
    torch.manual_seed(7)
    hidden = torch.randn(2, 4, 8)
    weight = torch.randn(6, 8)
    activation_projection = torch.randn(8, 3)
    weight_projection = torch.randn(6, 3)
    activation_error_projection = torch.randn(8, 3)
    a16 = pooled_probe_feature(
        hidden, weight, weight, activation_projection, weight_projection,
        "activation_error", activation_error_projection, activation_bits=16,
    )
    a4 = pooled_probe_feature(
        hidden, weight, weight, activation_projection, weight_projection,
        "activation_error", activation_error_projection, activation_bits=4,
    )
    assert a16.abs().max() == 0
    assert a4.abs().max() > 0

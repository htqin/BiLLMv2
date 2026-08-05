"""Calibration-only refinement with a strictly limited trainable surface."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


@dataclass(frozen=True)
class RefinementConfig:
    """Configure calibration-only compact-parameter refinement."""

    steps: int = 200
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    optimize: tuple[str, ...] = ("rotation", "scales", "low_rank")
    kl_weight: float = 0.1
    gradient_accumulation_steps: int = 1
    amp: bool = False
    rotation: str = "block_cayley"
    rank: int = 4
    block_size: int = 128


class RefinedLinear(nn.Module):
    """Wrap a frozen quantized linear with foldable compact trainable parameters."""

    def __init__(self, linear: nn.Linear, config: RefinementConfig) -> None:
        super().__init__()
        self.register_buffer("base_weight", linear.weight.detach().clone())
        self.register_buffer(
            "base_bias", linear.bias.detach().clone() if linear.bias is not None else None
        )
        self.activation_quantizer = getattr(
            linear, "_billmv2_activation_quantizer", None
        )
        rank = min(config.rank, linear.in_features, linear.out_features)
        self.scale = nn.Parameter(torch.ones(linear.out_features, 1))
        self.u = nn.Parameter(torch.zeros(linear.out_features, rank))
        self.v = nn.Parameter(torch.zeros(linear.in_features, rank))
        nn.init.normal_(self.v, std=1e-3)
        targets = set(config.optimize)
        self.activation_scale = None
        if (
            self.activation_quantizer is not None
            and self.activation_quantizer.bits < 16
            and "activation_scales" in targets
        ):
            self.activation_scale = nn.Parameter(
                self.activation_quantizer.scales.detach().float().clone()
            )
        self.optimize = frozenset(targets)
        self.rotation_mode = config.rotation if "rotation" in targets else "none"
        self.rotation_parameters = nn.ParameterList()
        for start in range(0, linear.in_features, config.block_size):
            width = min(config.block_size, linear.in_features - start)
            if self.rotation_mode == "block_cayley":
                self.rotation_parameters.append(nn.Parameter(torch.zeros(width, width)))
            elif self.rotation_mode == "householder":
                vector = torch.zeros(width)
                vector[0] = 1.0
                self.rotation_parameters.append(nn.Parameter(vector))
        self.scale.requires_grad_("scales" in targets)
        self.u.requires_grad_("low_rank" in targets)
        self.v.requires_grad_("low_rank" in targets)
        for parameter in self.rotation_parameters:
            parameter.requires_grad_("rotation" in targets)

    def _rotation(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        blocks = []
        for parameter in self.rotation_parameters:
            if self.rotation_mode == "block_cayley":
                skew = parameter - parameter.transpose(0, 1)
                identity = torch.eye(skew.shape[0], device=device, dtype=torch.float32)
                block = torch.linalg.solve(identity + skew.float(), identity - skew.float())
            else:
                vector = parameter.float()
                identity = torch.eye(vector.numel(), device=device, dtype=torch.float32)
                reflection = identity - 2 * torch.outer(vector, vector) / vector.square().sum().clamp_min(1e-8)
                basis = torch.zeros_like(vector)
                basis[0] = 1.0
                reference = identity - 2 * torch.outer(basis, basis)
                block = reflection @ reference
            blocks.append(block.to(dtype))
        if not blocks:
            return torch.eye(self.base_weight.shape[1], device=device, dtype=dtype)
        return torch.block_diag(*blocks)

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply compact refinements without exposing the frozen base weight."""

        if self.activation_quantizer is not None:
            if self.activation_scale is not None:
                self.activation_quantizer.scales = self.activation_scale
            inputs = self.activation_quantizer.fake_quant(inputs)
        rotation = self._rotation(inputs.device, torch.float32)
        weight = self.base_weight.float() * self.scale + self.u @ self.v.transpose(0, 1)
        bias = self.base_bias.float() if self.base_bias is not None else None
        output = functional.linear(inputs.float() @ rotation, weight, bias)
        return output.to(inputs.dtype)

    @torch.no_grad()
    def merged_linear(self) -> nn.Linear:
        """Fold all learned compact parameters into a plain linear."""

        rotation = self._rotation(self.base_weight.device, self.base_weight.dtype)
        weight = (
            self.base_weight * self.scale + self.u @ self.v.transpose(0, 1)
        ) @ rotation.transpose(0, 1)
        linear = nn.Linear(weight.shape[1], weight.shape[0], self.base_bias is not None).to(weight)
        linear.weight.copy_(weight)
        if self.base_bias is not None:
            linear.bias.copy_(self.base_bias)
        return linear


def freeze_and_wrap(
    model: nn.Module,
    config: RefinementConfig,
    target_names: set[str] | None = None,
) -> None:
    """Freeze a quantized model and expose only requested compact parameters."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and (target_names is None or name in target_names)
    ]
    for name, module in targets:
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, RefinedLinear(module, config))


def trainable_parameter_ratio(model: nn.Module) -> tuple[int, int, float]:
    """Return trainable count, total count, and ratio."""

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return trainable, total, trainable / total if total else 0.0


def export_refinement(model: nn.Module) -> dict[str, dict[str, object]]:
    """Export trainable compact parameters without frozen dense base weights."""

    payload: dict[str, dict[str, object]] = {}
    for name, module in model.named_modules():
        if not isinstance(module, RefinedLinear):
            continue
        values: dict[str, object] = {"rotation_mode": module.rotation_mode}
        if module.scale.requires_grad:
            values["scale"] = module.scale.detach().cpu()
        if module.u.requires_grad:
            values["u"] = module.u.detach().cpu()
            values["v"] = module.v.detach().cpu()
        if module.activation_scale is not None:
            values["activation_scales"] = module.activation_scale.detach().cpu()
        for index, parameter in enumerate(module.rotation_parameters):
            if parameter.requires_grad:
                values[f"rotation_{index}"] = parameter.detach().cpu()
        payload[name] = values
    return payload

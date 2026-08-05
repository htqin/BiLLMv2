"""Sketch-based branch functional low-rank regression and runtime adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class FunctionalFactors:
    """Represent a functional correction U @ V.T."""

    u: Tensor
    v: Tensor

    @property
    def rank(self) -> int:
        return self.u.shape[1]


@dataclass(frozen=True)
class FunctionalSpectrum:
    """Store one max-rank sketched regression and its nested SVD basis."""

    basis: Tensor
    left: Tensor
    singular: Tensor
    right: Tensor
    input_dim: int
    output_dim: int

    @property
    def max_rank(self) -> int:
        return int(self.singular.numel())


def truncate_functional_spectrum(spectrum: FunctionalSpectrum, rank: int, dtype_like: Tensor) -> FunctionalFactors:
    """Return nested factors from the leading components of one spectrum."""

    if rank < 0 or rank > spectrum.max_rank:
        raise ValueError("rank must be within the fitted spectrum")
    if rank == 0:
        return FunctionalFactors(
            dtype_like.new_zeros((spectrum.output_dim, 0)),
            dtype_like.new_zeros((spectrum.input_dim, 0)),
        )
    root = spectrum.singular[:rank].sqrt()
    u = spectrum.left[:, :rank] * root.unsqueeze(0)
    v = (spectrum.basis @ spectrum.right[:rank].transpose(0, 1)) * root.unsqueeze(0)
    return FunctionalFactors(u.to(dtype_like), v.to(dtype_like))


def functional_energy_explained(spectrum: FunctionalSpectrum, rank: int) -> float:
    """Return cumulative regression energy explained by the leading rank."""

    if spectrum.singular.numel() == 0:
        return 0.0
    total = spectrum.singular.square().sum().clamp_min(1e-12)
    used = spectrum.singular[:rank].square().sum()
    return float((used / total).clamp(0, 1))


@torch.no_grad()
def fit_functional_spectrum(
    inputs: Tensor,
    residual: Tensor,
    max_rank: int,
    ridge: float = 1e-4,
    sketch_dim: int = 128,
    max_tokens: int = 4096,
    seed: int = 0,
) -> FunctionalSpectrum:
    """Fit one max-rank sketched ridge regression and keep its SVD spectrum."""

    z = inputs.detach().float().reshape(-1, inputs.shape[-1])
    target = residual.detach().float().reshape(-1, residual.shape[-1])
    if z.shape[0] != target.shape[0] or max_rank < 0 or ridge < 0:
        raise ValueError("functional regression inputs are incompatible")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if z.shape[0] > max_tokens:
        indices = torch.randperm(z.shape[0], generator=generator)[:max_tokens].to(z.device)
        z = z[indices]
        target = target[indices]
    dimension = min(sketch_dim, z.shape[0], z.shape[1])
    if max_rank == 0 or dimension == 0:
        empty = z.new_zeros((0, 0))
        return FunctionalSpectrum(
            z.new_zeros((z.shape[1], 0)), z.new_zeros((target.shape[1], 0)),
            z.new_zeros((0,)), empty, z.shape[1], target.shape[1],
        )
    omega = torch.randn((z.shape[0], dimension), generator=generator).to(z.device)
    basis, _ = torch.linalg.qr(z.transpose(0, 1) @ omega, mode="reduced")
    basis, _ = torch.linalg.qr(z.transpose(0, 1) @ (z @ basis), mode="reduced")
    reduced_inputs = z @ basis
    gram = reduced_inputs.transpose(0, 1) @ reduced_inputs
    gram.diagonal().add_(ridge)
    regression = torch.linalg.solve(
        gram, reduced_inputs.transpose(0, 1) @ target
    ).transpose(0, 1)
    left, singular, right = torch.linalg.svd(regression, full_matrices=False)
    selected = min(max_rank, singular.numel())
    return FunctionalSpectrum(
        basis[:, : right.shape[1]], left[:, :selected], singular[:selected],
        right[:selected], z.shape[1], target.shape[1],
    )


@torch.no_grad()
def fit_functional_low_rank(
    inputs: Tensor,
    residual: Tensor,
    rank: int,
    ridge: float = 1e-4,
    sketch_dim: int = 128,
    max_tokens: int = 4096,
    seed: int = 0,
) -> FunctionalFactors:
    """Fit R approximately Z V U.T using sketched ridge regression."""

    spectrum = fit_functional_spectrum(
        inputs, residual, rank, ridge, sketch_dim, max_tokens, seed
    )
    return truncate_functional_spectrum(spectrum, min(rank, spectrum.max_rank), inputs)


def _quantize_int8_factor(
    factor: Tensor, scale_mode: str
) -> tuple[Tensor, Tensor, Tensor]:
    if scale_mode == "tensor":
        scale = (factor.abs().max().clamp_min(1e-12) / 127).half()
        quantized = torch.round(factor / scale.float()).clamp(-127, 127).to(torch.int8)
        runtime = quantized.to(factor.dtype) * scale.to(factor.dtype)
        return quantized, scale, runtime
    if scale_mode != "per_rank":
        raise ValueError(f"unsupported int8 scale mode: {scale_mode}")
    scale = (factor.abs().amax(dim=0).clamp_min(1e-12) / 127).half()
    quantized = torch.round(factor / scale.float().unsqueeze(0)).clamp(-127, 127).to(torch.int8)
    runtime = quantized.to(factor.dtype) * scale.to(factor.dtype).unsqueeze(0)
    return quantized, scale, runtime


def functional_factor_payload(
    factors: FunctionalFactors,
    dtype: str,
    target_module: str,
    int8_scale_mode: str = "tensor",
    metadata: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Tensor, Tensor]:
    """Encode functional factors and return their dequantized runtime form."""

    if factors.rank == 0:
        return {}, factors.u, factors.v
    payload: dict[str, Any] = {
        "mode": "functional_branch", "rank": factors.rank,
        "target_module": target_module, "schema_version": 2,
    }
    if metadata:
        payload.update(metadata)
    if dtype == "fp16":
        stored_u, stored_v = factors.u.half(), factors.v.half()
        payload.update({"u": stored_u.cpu(), "v": stored_v.cpu()})
        return payload, stored_u.to(factors.u), stored_v.to(factors.v)
    if dtype != "int8":
        raise ValueError(f"unsupported functional factor dtype: {dtype}")
    payload["int8_scale_mode"] = int8_scale_mode
    runtime = []
    for name, factor in (("u", factors.u), ("v", factors.v)):
        quantized, scale, dequantized = _quantize_int8_factor(factor, int8_scale_mode)
        payload[name] = quantized.cpu()
        payload[f"{name}_scale"] = scale.cpu()
        runtime.append(dequantized)
    return payload, runtime[0], runtime[1]


def decode_functional_factors(
    payload: dict[str, Any], device: torch.device
) -> tuple[Tensor, Tensor]:
    """Decode stored functional factors for inference."""

    u, v = payload["u"].to(device), payload["v"].to(device)
    if u.dtype == torch.int8:
        mode = payload.get("int8_scale_mode", "tensor")
        u_scale = payload["u_scale"].to(device).float()
        v_scale = payload["v_scale"].to(device).float()
        if mode == "per_rank":
            u = u.float() * u_scale.unsqueeze(0)
            v = v.float() * v_scale.unsqueeze(0)
        elif mode == "tensor":
            u = u.float() * u_scale
            v = v.float() * v_scale
        else:
            raise ValueError(f"unsupported int8 scale mode: {mode}")
    return u.float(), v.float()


def remove_functional_adapter(module: nn.Module) -> None:
    """Remove a previously installed functional correction hook."""

    handle = getattr(module, "_billmv2_functional_low_rank_hook", None)
    if handle is not None:
        handle.remove()
        delattr(module, "_billmv2_functional_low_rank_hook")
    for name in ("_billmv2_functional_u", "_billmv2_functional_v"):
        if hasattr(module, name):
            delattr(module, name)


def install_functional_adapter(module: nn.Linear, payload: dict[str, Any]) -> None:
    """Install Y = XQ.T + (XV)U.T without merging dense weights."""

    remove_functional_adapter(module)
    if not payload or payload.get("mode") != "functional_branch":
        return
    u, v = decode_functional_factors(payload, module.weight.device)
    module.register_buffer("_billmv2_functional_u", u.to(module.weight.dtype))
    module.register_buffer("_billmv2_functional_v", v.to(module.weight.dtype))

    def add_correction(_module: nn.Module, arguments: tuple[Tensor, ...], output: Tensor) -> Tensor:
        inputs = arguments[0]
        correction = (inputs.float() @ _module._billmv2_functional_v.float()) @ (
            _module._billmv2_functional_u.float().transpose(0, 1)
        )
        return output + correction.to(output.dtype)

    module._billmv2_functional_low_rank_hook = module.register_forward_hook(add_correction)


def functional_parameter_bits(
    in_features: int,
    out_features: int,
    rank: int,
    dtype: str,
    int8_scale_mode: str = "tensor",
) -> int:
    """Return exact factor payload bits excluding container metadata."""

    if rank == 0:
        return 0
    value_bits = 8 if dtype == "int8" else 16
    if dtype == "int8":
        scale_bits = 32 if int8_scale_mode == "tensor" else 32 * rank
    else:
        scale_bits = 0
    return value_bits * rank * (in_features + out_features) + scale_bits

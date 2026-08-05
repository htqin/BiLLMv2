"""Low-dimensional calibration feature extraction."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn

from billmv2.quantization.activation import ActivationQuantizer
from billmv2.quantization.binarizer import binary_approximation


class RandomProjector:
    """Project activation and provisional-error summaries without retaining tokens."""

    def __init__(self, input_dim: int, output_dim: int = 32, seed: int = 0) -> None:
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("projection dimensions must be positive")
        generator = torch.Generator().manual_seed(seed)
        self.matrix = torch.randn(
            input_dim, output_dim, generator=generator, dtype=torch.float32
        ) / output_dim**0.5

    def sequence_feature(self, hidden: Tensor, error_hidden: Tensor | None = None) -> Tensor:
        """Return a sequence-level projected mean and error statistic."""

        if hidden.shape[-1] != self.matrix.shape[0]:
            raise ValueError("hidden dimension does not match projector")
        summary = hidden.detach().float().reshape(-1, hidden.shape[-1]).mean(0).cpu()
        features = [summary @ self.matrix]
        if error_hidden is not None:
            if error_hidden.shape[-1] != self.matrix.shape[0]:
                raise ValueError("error hidden dimension does not match projector")
            error = error_hidden.detach().float().reshape(-1, error_hidden.shape[-1]).mean(0).cpu()
            features.append(error @ self.matrix)
        return torch.cat(features)


def probe_layer_indices(num_layers: int, stride: int = 4) -> list[int]:
    """Choose sparse probe layers, always including the final block."""

    if num_layers <= 0 or stride <= 0:
        raise ValueError("num_layers and stride must be positive")
    indices = list(range(0, num_layers, stride))
    if indices[-1] != num_layers - 1:
        indices.append(num_layers - 1)
    return indices


def provisional_binary_weight(weight: Tensor) -> Tensor:
    """Build a lightweight one-term provisional binary approximation."""

    diagonal = torch.ones(weight.shape[1], device=weight.device)
    return binary_approximation(weight, hessian_diag=diagonal, order=1).quantized


def pooled_probe_feature(
    hidden: Tensor,
    weight: Tensor,
    provisional: Tensor,
    activation_projection: Tensor,
    error_projection: Tensor,
    mode: str = "joint",
    activation_error_projection: Tensor | None = None,
    activation_bits: int = 4,
    activation_group_size: int = 128,
    activation_symmetric: bool = True,
    activation_clip_method: str = "mse",
    rotation: str = "none",
    rotation_block_size: int = 128,
    rotation_seed: int = 0,
) -> Tensor:
    """Project pooled activation and functional binary error without retaining tokens."""

    if weight.shape != provisional.shape or hidden.shape[-1] != weight.shape[1]:
        raise ValueError("incompatible hidden, weight, or provisional shapes")
    summary = hidden.detach().reshape(-1, hidden.shape[-1]).float().mean(0)
    components = []
    if mode in {"activation", "joint"}:
        components.append(summary @ activation_projection.float())
    if mode in {"binary_error", "weight_error", "joint"}:
        functional_error = summary.to(weight.dtype) @ (weight - provisional).transpose(0, 1)
        components.append(functional_error.float() @ error_projection.float())
    if mode in {"activation_error", "joint"}:
        projection = (
            activation_projection
            if activation_error_projection is None
            else activation_error_projection
        )
        activation_quantizer = ActivationQuantizer(
            activation_bits, activation_group_size, activation_symmetric,
            activation_clip_method, rotation, rotation_block_size, rotation_seed,
        )
        activation_quantizer.observe(hidden)
        activation_quantizer.finalize()
        activation_error = (
            hidden.detach().float() - activation_quantizer.fake_quant(hidden).float()
        ).reshape(-1, hidden.shape[-1]).mean(0)
        components.append(activation_error @ projection.float())
    if not components:
        raise ValueError(f"unsupported probe feature mode: {mode}")
    return torch.cat(components).cpu()


def _probe_modules(model: nn.Module, stride: int) -> list[tuple[str, nn.Linear]]:
    layers = model.model.decoder.layers if model.config.model_type == "opt" else model.model.layers
    probes = []
    for layer_index in probe_layer_indices(len(layers), stride):
        layer = layers[layer_index]
        candidates = (
            ("self_attn.q_proj", getattr(getattr(layer, "self_attn", None), "q_proj", None)),
            ("mlp.down_proj", getattr(getattr(layer, "mlp", None), "down_proj", None)),
            ("fc2", getattr(layer, "fc2", None)),
        )
        attention_added = False
        mlp_added = False
        for suffix, module in candidates:
            if not isinstance(module, nn.Linear):
                continue
            is_attention = suffix.startswith("self_attn")
            if (is_attention and attention_added) or (not is_attention and mlp_added):
                continue
            probes.append((f"{layer_index}.{suffix}", module))
            attention_added |= is_attention
            mlp_added |= not is_attention
    return probes


@torch.no_grad()
def extract_calibration_features(
    model: nn.Module,
    samples: list[object],
    device: str | torch.device,
    feature_dim: int = 64,
    probe_stride: int = 4,
    mode: str = "joint",
    seed: int = 0,
    provisional_fn: Callable[[Tensor], Tensor] = provisional_binary_weight,
    activation_bits: int = 4,
    activation_group_size: int = 128,
    activation_symmetric: bool = True,
    activation_clip_method: str = "mse",
    rotation: str = "none",
    rotation_block_size: int = 128,
) -> Tensor:
    """Extract deterministic sparse-probe features and remove every hook on exit."""

    if not samples or feature_dim <= 0:
        raise ValueError("samples and feature_dim must be non-empty and positive")
    target = torch.device(device)
    model.to(target).eval()
    probes = _probe_modules(model, probe_stride)
    if not probes:
        raise ValueError("no supported attention or MLP probe modules found")
    generator = torch.Generator().manual_seed(seed)
    specifications = []
    for name, module in probes:
        weight = module.weight.detach()
        provisional = provisional_fn(weight)
        activation_projection = torch.randn(
            module.in_features, feature_dim, generator=generator
        ).to(target) / feature_dim**0.5
        error_projection = torch.randn(
            module.out_features, feature_dim, generator=generator
        ).to(target) / feature_dim**0.5
        activation_error_projection = torch.randn(
            module.in_features, feature_dim, generator=generator
        ).to(target) / feature_dim**0.5
        specifications.append(
            (name, module, weight, provisional, activation_projection,
             error_projection, activation_error_projection)
        )
    rows = []
    try:
        for sample in samples:
            captured: dict[str, Tensor] = {}
            handles = []
            for (name, module, weight, provisional, activation_projection,
                 error_projection, activation_error_projection) in specifications:
                def capture(
                    _module: nn.Module,
                    arguments: tuple[Tensor, ...],
                    probe_name: str = name,
                    probe_weight: Tensor = weight,
                    probe_provisional: Tensor = provisional,
                    projection_h: Tensor = activation_projection,
                    projection_e: Tensor = error_projection,
                    projection_a: Tensor = activation_error_projection,
                ) -> None:
                    captured[probe_name] = pooled_probe_feature(
                        arguments[0], probe_weight, probe_provisional,
                        projection_h, projection_e, mode, projection_a,
                        activation_bits, activation_group_size,
                        activation_symmetric, activation_clip_method,
                        rotation, rotation_block_size, seed,
                    )
                handles.append(module.register_forward_pre_hook(capture))
            try:
                token_ids = sample[0] if isinstance(sample, (tuple, list)) else sample
                model(token_ids.to(target), use_cache=False)
            finally:
                for handle in handles:
                    handle.remove()
            rows.append(torch.cat([captured[name] for name, *_ in specifications]))
    finally:
        model.cpu()
        specifications.clear()
        torch.cuda.empty_cache()
    stacked = torch.stack(rows)
    if stacked.shape[1] != feature_dim:
        final_generator = torch.Generator().manual_seed(seed + 104729)
        final_projection = torch.randn(
            stacked.shape[1], feature_dim, generator=final_generator, dtype=torch.float32
        ) / feature_dim**0.5
        stacked = stacked.float() @ final_projection
    return stacked

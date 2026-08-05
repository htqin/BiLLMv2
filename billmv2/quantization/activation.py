"""Reference group-wise activation quantization."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from billmv2.transforms.rotation import apply_block_rotation


class ActivationQuantizer:
    """Calibrate and apply static group-wise activation fake quantization."""

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        symmetric: bool = True,
        clip_method: str = "mse",
        rotation: str = "none",
        rotation_block_size: int = 128,
        rotation_seed: int = 0,
    ) -> None:
        if bits not in {4, 8, 16}:
            raise ValueError("activation bits must be 4, 8, or 16")
        if group_size <= 0:
            raise ValueError("activation group size must be positive")
        if clip_method not in {"max", "percentile", "mse"}:
            raise ValueError(f"unsupported activation clip method: {clip_method}")
        self.bits = bits
        self.group_size = group_size
        self.symmetric = symmetric
        self.clip_method = clip_method
        self.rotation = rotation
        self.rotation_block_size = rotation_block_size
        self.rotation_seed = rotation_seed
        self.feature_dim: int | None = None
        self._samples: list[Tensor] = []
        self.scales: Tensor | None = None
        self.zero_points: Tensor | None = None

    def observe(self, inputs: Tensor) -> None:
        """Accumulate a bounded calibration sample without retaining full activations."""

        if inputs.ndim < 2:
            raise ValueError("activation tensor must have a feature dimension")
        width = inputs.shape[-1]
        if self.feature_dim is not None and self.feature_dim != width:
            raise ValueError("activation feature dimension changed during calibration")
        self.feature_dim = width
        if self.bits == 16:
            return
        rotated = apply_block_rotation(
            inputs.detach(), self.rotation, self.rotation_block_size, self.rotation_seed
        )
        flat = rotated.float().reshape(-1, width).cpu()
        retained = sum(sample.shape[0] for sample in self._samples)
        if retained < 4096:
            self._samples.append(flat[: min(flat.shape[0], 4096 - retained)])

    def _grouped_samples(self) -> tuple[Tensor, int]:
        if self.feature_dim is None or not self._samples:
            raise RuntimeError("activation quantizer has no calibration samples")
        values = torch.cat(self._samples)
        groups = math.ceil(self.feature_dim / self.group_size)
        padded_width = groups * self.group_size
        if padded_width != self.feature_dim:
            values = torch.nn.functional.pad(values, (0, padded_width - self.feature_dim))
        return values.reshape(values.shape[0], groups, self.group_size), groups

    def finalize(self) -> None:
        """Determine one persistent scale per input-feature group."""

        if self.bits == 16:
            self.scales = torch.empty(0, dtype=torch.float16)
            self.zero_points = torch.empty(0, dtype=torch.int8)
            self._samples.clear()
            return
        grouped, _ = self._grouped_samples()
        if self.symmetric:
            qmin, qmax = -(2 ** (self.bits - 1)), 2 ** (self.bits - 1) - 1
            absolute = grouped.abs()
            if self.clip_method == "max":
                clip = absolute.amax(dim=(0, 2))
            elif self.clip_method == "percentile":
                clip = torch.quantile(absolute.permute(1, 0, 2).flatten(1), 0.999, dim=1)
            else:
                maximum = absolute.amax(dim=(0, 2))
                candidates = torch.linspace(0.5, 1.0, 11).unsqueeze(1) * maximum.unsqueeze(0)
                errors = []
                for candidate in candidates:
                    scale = (candidate / qmax).clamp_min(torch.finfo(torch.float32).eps)
                    quantized = torch.clamp(
                        torch.round(grouped / scale.view(1, -1, 1)), qmin, qmax
                    ) * scale.view(1, -1, 1)
                    errors.append((grouped - quantized).square().mean(dim=(0, 2)))
                stacked = torch.stack(errors)
                clip = candidates.gather(0, stacked.argmin(0).unsqueeze(0)).squeeze(0)
            self.scales = (clip / qmax).clamp_min(torch.finfo(torch.float16).tiny).half()
            self.zero_points = torch.zeros_like(self.scales, dtype=torch.int8)
        else:
            qmin, qmax = 0, 2**self.bits - 1
            minimum = grouped.amin(dim=(0, 2))
            maximum = grouped.amax(dim=(0, 2))
            scale = ((maximum - minimum) / qmax).clamp_min(torch.finfo(torch.float16).tiny)
            self.scales = scale.half()
            self.zero_points = torch.round(-minimum / scale).clamp(qmin, qmax).to(torch.int16)
        self._samples.clear()

    def fake_quant(self, inputs: Tensor) -> Tensor:
        """Apply the reference quantize-dequantize path in the input dtype."""

        if self.bits == 16:
            return inputs
        if self.scales is None or self.zero_points is None or self.feature_dim != inputs.shape[-1]:
            raise RuntimeError("activation quantizer is not calibrated for this input")
        groups = self.scales.numel()
        padded_width = groups * self.group_size
        rotated_inputs = apply_block_rotation(
            inputs, self.rotation, self.rotation_block_size, self.rotation_seed
        )
        values = rotated_inputs.float()
        if padded_width != self.feature_dim:
            values = torch.nn.functional.pad(values, (0, padded_width - self.feature_dim))
        shape = values.shape
        grouped = values.reshape(*shape[:-1], groups, self.group_size)
        scales = self.scales.to(inputs.device).float()
        view_shape = (1,) * (grouped.ndim - 2) + (groups, 1)
        scales = scales.view(view_shape)
        if self.symmetric:
            qmin, qmax = -(2 ** (self.bits - 1)), 2 ** (self.bits - 1) - 1
            quantized = torch.round(grouped / scales).clamp(qmin, qmax) * scales
        else:
            qmin, qmax = 0, 2**self.bits - 1
            zero_points = self.zero_points.to(inputs.device).float().view(view_shape)
            integers = torch.round(grouped / scales + zero_points).clamp(qmin, qmax)
            quantized = (integers - zero_points) * scales
        quantized = quantized.reshape(*shape)[..., : self.feature_dim].to(inputs.dtype)
        return apply_block_rotation(
            quantized, self.rotation, self.rotation_block_size,
            self.rotation_seed, inverse=True,
        )

    def to_artifact(self) -> dict[str, Any]:
        """Return persistent runtime quantization parameters."""

        if self.scales is None or self.zero_points is None:
            raise RuntimeError("activation quantizer must be finalized")
        payload: dict[str, Any] = {
            "bits": self.bits,
            "group_size": self.group_size,
            "symmetric": self.symmetric,
            "clip_method": self.clip_method,
            "feature_dim": self.feature_dim,
            "rotation": self.rotation,
            "rotation_block_size": self.rotation_block_size,
            "rotation_seed": self.rotation_seed,
        }
        if self.bits < 16:
            payload["scales"] = self.scales.cpu()
            if not self.symmetric:
                payload["zero_points"] = self.zero_points.cpu()
        return payload

    @classmethod
    def from_artifact(cls, payload: dict[str, Any]) -> "ActivationQuantizer":
        """Restore runtime activation quantization parameters."""

        quantizer = cls(
            int(payload["bits"]),
            int(payload["group_size"]),
            bool(payload["symmetric"]),
            str(payload["clip_method"]),
            str(payload.get("rotation", "none")),
            int(payload.get("rotation_block_size", 128)),
            int(payload.get("rotation_seed", 0)),
        )
        quantizer.feature_dim = int(payload["feature_dim"])
        quantizer.scales = payload.get("scales", torch.empty(0, dtype=torch.float16))
        quantizer.zero_points = payload.get(
            "zero_points", torch.zeros_like(quantizer.scales, dtype=torch.int8)
        )
        return quantizer


def install_activation_quantizer(module: torch.nn.Module, payload: dict[str, Any]) -> None:
    """Install one persistent input fake-quant hook on a Linear-like module."""

    previous = getattr(module, "_billmv2_activation_hook", None)
    if previous is not None:
        previous.remove()
    quantizer = ActivationQuantizer.from_artifact(payload)
    module._billmv2_activation_quantizer = quantizer
    module._billmv2_activation_hook = module.register_forward_pre_hook(
        lambda _module, arguments: (quantizer.fake_quant(arguments[0]),)
    )

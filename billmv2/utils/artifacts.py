"""Compact artifact persistence and reconstruction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from billmv2.transforms.rotation import make_block_rotation
from billmv2.quantization.activation import install_activation_quantizer
from billmv2.low_rank.functional import install_functional_adapter
from billmv2.utils.bits import account_saved_artifact_storage, unpack_bits, unpack_indices

ARTIFACT_FORMAT_VERSION = 5
SUPPORTED_ARTIFACT_FORMAT_VERSIONS = {3, 4, ARTIFACT_FORMAT_VERSION}


def _low_rank(
    payload: dict[str, Tensor], device: torch.device, shape: tuple[int, int]
) -> Tensor:
    if not payload:
        return torch.zeros(shape, device=device)
    if payload.get("mode") == "functional_branch":
        return torch.zeros(shape, device=device)
    u = payload["u"].to(device)
    v = payload["v"].to(device)
    if u.dtype == torch.int8:
        u = u.float() * payload["u_scale"].to(device)
        v = v.float() * payload["v_scale"].to(device)
    return u.float() @ v.float().transpose(0, 1)


def _reconstruct_branch(
    sign_payload: dict[str, object],
    scales: Tensor,
    mask: Tensor,
    means: Tensor | None = None,
) -> Tensor:
    """Reconstruct one masked binary branch from active-only signs."""

    rows, width = mask.shape
    active = unpack_bits(sign_payload, mask.device).float().mul_(2).sub_(1)
    signs = torch.zeros((active.shape[0], rows, width), device=mask.device)
    signs[:, mask] = active
    values = signs * scales.float().transpose(0, 1).unsqueeze(-1)
    if means is not None:
        values += means.float().transpose(0, 1).unsqueeze(-1) * mask
    return values.sum(0)


def reconstruct_weight(artifact: dict[str, Any], device: torch.device | str = "cpu") -> Tensor:
    """Reconstruct a quantized weight from compact binary artifacts."""

    target = torch.device(device)
    rows, columns = artifact["shape"]
    rotated = torch.zeros((rows, columns), device=target)
    low_rank = _low_rank(artifact["low_rank"], target, (rows, columns))
    for block in artifact["blocks"]:
        width = block["end"] - block["start"]
        salient_mask = torch.zeros((rows, width), dtype=torch.bool, device=target)
        stored_indices = block["salient_indices"]
        salient_indices = (
            stored_indices.to(target).long()
            if isinstance(stored_indices, Tensor)
            else unpack_indices(stored_indices, target)
        )
        salient_mask[:, salient_indices] = True
        sparse_mask = unpack_bits(block["sparse_mask"], target)
        concentrated_mask = ~salient_mask & ~sparse_mask
        core = torch.zeros((rows, width), device=target)
        for name, mask in (("salient", salient_mask), ("concentrated", concentrated_mask), ("sparse", sparse_mask)):
            core += _reconstruct_branch(
                block[f"{name}_signs"],
                block[f"{name}_scales"].to(target),
                mask,
                (
                    block[f"{name}_means"].to(target)
                    if isinstance(block.get(f"{name}_means"), Tensor)
                    else None
                ),
            )
        core += low_rank[:, block["start"] : block["end"]]
        rotated[:, block["start"] : block["end"]] = core
    rotation = artifact.get("rotation_matrix")
    if rotation is None and artifact.get("rotation", "none") != "none":
        rotation = make_block_rotation(
            columns,
            artifact["rotation"],
            artifact["rotation_block_size"],
            artifact["rotation_seed"],
            target,
        )
    if rotation is not None:
        rotated = rotated @ rotation.to(target).float().transpose(0, 1)
    return rotated


def apply_artifacts(model: nn.Module, artifacts: dict[str, Any]) -> None:
    """Apply compact layer artifacts to a freshly loaded full-precision model."""

    modules = dict(model.named_modules())
    for name, artifact in artifacts.items():
        module = modules.get(name)
        if not isinstance(module, nn.Linear):
            raise KeyError(f"artifact target is not a linear layer: {name}")
        module.weight.data.copy_(reconstruct_weight(artifact, module.weight.device).to(module.weight))
        install_activation_quantizer(module, artifact["activation"])
        install_functional_adapter(module, artifact["low_rank"])


def save_run_artifacts(
    output_dir: Path,
    config: dict[str, Any],
    binary_artifacts: dict[str, Any],
    metrics: dict[str, Any],
    bpw: dict[str, Any],
    calibration: dict[str, Any],
) -> None:
    """Write compact run outputs without serializing a full model."""

    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        **config,
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "artifact_type": "ptq",
    }
    for name, value in (
        ("config.json", config),
        ("metrics.json", metrics),
        ("calibration.json", calibration),
    ):
        (output_dir / name).write_text(json.dumps(value, indent=2), encoding="utf-8")
    binary_only = {}
    split_diagnostics: dict[str, Any] = {}
    for name, artifact in binary_artifacts.items():
        cleaned = {key: value for key, value in artifact.items() if key != "low_rank"}
        block_diagnostics = []
        for block in cleaned.get("blocks", []):
            diagnostic = block.pop("split_diagnostics", None)
            if diagnostic is not None:
                block_diagnostics.append({
                    "start": int(block["start"]),
                    "end": int(block["end"]),
                    **diagnostic,
                })
        if block_diagnostics:
            split_diagnostics[name] = block_diagnostics
        binary_only[name] = cleaned
    torch.save(binary_only, output_dir / "binary_artifacts.pt")
    low_rank = {name: artifact["low_rank"] for name, artifact in binary_artifacts.items()}
    rotations = {
        name: artifact["rotation_matrix"]
        for name, artifact in binary_artifacts.items()
        if artifact["rotation_matrix"] is not None
    }
    torch.save(low_rank, output_dir / "low_rank.pt")
    torch.save(rotations, output_dir / "rotations.pt")
    if split_diagnostics:
        (output_dir / "split_diagnostics.json").write_text(
            json.dumps(split_diagnostics, indent=2), encoding="utf-8"
        )
    measured_bpw = account_saved_artifact_storage(bpw, output_dir)
    (output_dir / "bpw.json").write_text(
        json.dumps(measured_bpw, indent=2), encoding="utf-8"
    )


def load_billmv2_artifacts(
    artifact_dir: Path | str,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load and validate a versioned PTQ or PTQ+ artifact bundle."""

    directory = Path(artifact_dir)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    version = config.get("artifact_format_version")
    if version not in SUPPORTED_ARTIFACT_FORMAT_VERSIONS:
        raise ValueError(
            f"unsupported artifact format version: {version}; expected one of {sorted(SUPPORTED_ARTIFACT_FORMAT_VERSIONS)}"
        )
    binary = torch.load(directory / "binary_artifacts.pt", map_location=map_location)
    low_rank = torch.load(directory / "low_rank.pt", map_location=map_location)
    if set(binary) != set(low_rank):
        raise ValueError("binary and low-rank artifact layer sets do not match")
    merged = {name: {**artifact, "low_rank": low_rank[name]} for name, artifact in binary.items()}
    artifact_type = config.get("artifact_type", "ptq")
    if artifact_type not in {"ptq", "ptq_ft"}:
        raise ValueError(f"unsupported artifact type: {artifact_type}")
    refinement_path = directory / "refinement.pt"
    if artifact_type == "ptq_ft" and not refinement_path.exists():
        raise FileNotFoundError("PTQ+ artifact is missing refinement.pt")
    refinement = (
        torch.load(refinement_path, map_location=map_location)
        if artifact_type == "ptq_ft"
        else None
    )
    return {
        "format_version": version,
        "config": config,
        "artifacts": merged,
        "refinement": refinement,
    }


def _refinement_rotation(
    payload: dict[str, Any], device: torch.device, width: int
) -> Tensor:
    blocks = []
    mode = payload["rotation_mode"]
    rotation_keys = sorted(
        (key for key in payload if key.startswith("rotation_") and key.rsplit("_", 1)[1].isdigit()),
        key=lambda key: int(key.rsplit("_", 1)[1]),
    )
    for key in rotation_keys:
        parameter = payload[key].to(device).float()
        identity = torch.eye(parameter.shape[0], device=device)
        if mode == "block_cayley":
            skew = parameter - parameter.transpose(0, 1)
            block = torch.linalg.solve(identity + skew, identity - skew)
        elif mode == "householder":
            reflection = identity - 2 * torch.outer(parameter, parameter) / parameter.square().sum().clamp_min(1e-8)
            basis = torch.zeros_like(parameter)
            basis[0] = 1.0
            reference = identity - 2 * torch.outer(basis, basis)
            block = reflection @ reference
        else:
            raise ValueError(f"unsupported refinement rotation: {mode}")
        blocks.append(block)
    if not blocks:
        return torch.eye(width, device=device)
    return torch.block_diag(*blocks)


@torch.no_grad()
def apply_billmv2_artifacts(
    model: nn.Module,
    bundle: dict[str, Any],
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> nn.Module:
    """Apply PTQ then optional compact refinement and return an evaluable model."""

    if bundle.get("format_version") not in SUPPORTED_ARTIFACT_FORMAT_VERSIONS:
        raise ValueError("artifact bundle has an unsupported format version")
    apply_artifacts(model, bundle["artifacts"])
    refinement = bundle.get("refinement")
    if refinement is not None:
        modules = dict(model.named_modules())
        for name, payload in refinement.items():
            module = modules.get(name)
            if not isinstance(module, nn.Linear):
                raise KeyError(f"refinement target is not a linear layer: {name}")
            target = module.weight.device
            activation_scales = payload.get("activation_scales")
            if isinstance(activation_scales, Tensor):
                activation_quantizer = getattr(
                    module, "_billmv2_activation_quantizer", None
                )
                if activation_quantizer is None:
                    raise ValueError(f"activation refinement without A4 payload: {name}")
                activation_quantizer.scales = activation_scales.to(target)
            scale_value = payload.get("scale")
            scale = (
                scale_value.to(target).float()
                if isinstance(scale_value, Tensor)
                else torch.ones((module.out_features, 1), device=target)
            )
            if scale.shape != (module.out_features, 1):
                raise ValueError(f"invalid refinement scale shape for {name}")
            u_value = payload.get("u")
            v_value = payload.get("v")
            if isinstance(u_value, Tensor) and isinstance(v_value, Tensor):
                u = u_value.to(target).float()
                v = v_value.to(target).float()
                if u.shape[0] != module.out_features or v.shape[0] != module.in_features:
                    raise ValueError(f"invalid refinement factor shape for {name}")
                low_rank = u @ v.transpose(0, 1)
            elif u_value is None and v_value is None:
                low_rank = torch.zeros_like(module.weight, dtype=torch.float32)
            else:
                raise ValueError(f"incomplete refinement factors for {name}")
            rotation = _refinement_rotation(payload, target, module.in_features)
            merged = (module.weight.float() * scale + low_rank) @ rotation.transpose(0, 1)
            module.weight.copy_(merged.to(module.weight.dtype))
    if dtype is not None:
        model.to(dtype=dtype)
    if device is not None:
        model.to(device=device)
    return model

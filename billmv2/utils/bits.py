"""Exact storage accounting for compact quantization artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class BitCount:
    """Break down storage used by a quantized tensor."""

    num_weights: int
    binary_signs: int
    masks: int = 0
    scales: int = 0
    means: int = 0
    thresholds: int = 0
    rotations: int = 0
    low_rank: int = 0
    metadata: int = 0

    @property
    def strict_binary_bpw(self) -> float:
        """Return bits per original weight for binary signs only."""

        return self.binary_signs / self.num_weights if self.num_weights else 0.0

    @property
    def total_bits(self) -> int:
        """Return all stored bits."""

        return sum(
            (
                self.binary_signs,
                self.masks,
                self.scales,
                self.means,
                self.thresholds,
                self.rotations,
                self.low_rank,
                self.metadata,
            )
        )

    @property
    def effective_bpw(self) -> float:
        """Return total effective bits per original weight."""

        return self.total_bits / self.num_weights if self.num_weights else 0.0


def tensor_storage_bits(numel: int, dtype_bits: int) -> int:
    """Return storage bits for a dense tensor."""

    if numel < 0 or dtype_bits <= 0:
        raise ValueError("numel must be non-negative and dtype_bits positive")
    return numel * dtype_bits


def pack_bits(tensor: Tensor) -> dict[str, object]:
    """Pack a Boolean/sign tensor into bytes with reconstruction metadata."""

    boolean = tensor.detach().cpu().bool().reshape(-1).numpy()
    data = torch.from_numpy(np.packbits(boolean, bitorder="little").copy())
    return {
        "data": data,
        "shape": list(tensor.shape),
        "numel": tensor.numel(),
        "true_count": int(boolean.sum()),
    }


def unpack_bits(payload: dict[str, object], device: torch.device | str = "cpu") -> Tensor:
    """Unpack a tensor created by :func:`pack_bits`."""

    data = payload["data"]
    if not isinstance(data, Tensor):
        raise TypeError("packed bit data must be a tensor")
    array = np.unpackbits(data.cpu().numpy(), bitorder="little")[: int(payload["numel"])]
    return torch.from_numpy(array.copy()).reshape(payload["shape"]).bool().to(device)


def pack_indices(indices: Tensor, width: int) -> dict[str, object]:
    """Encode sorted structured indices with a combinatorial rank."""

    values = sorted(int(value) for value in indices.detach().cpu().tolist())
    if len(set(values)) != len(values) or any(value < 0 or value >= width for value in values):
        raise ValueError("indices must be unique and within the block width")
    rank = sum(math.comb(value, order) for order, value in enumerate(values, start=1))
    logical_bits = max(1, (math.comb(width, len(values)) - 1).bit_length()) if values else 0
    data = torch.tensor(
        list(rank.to_bytes((logical_bits + 7) // 8, byteorder="little")), dtype=torch.uint8
    )
    return {"data": data, "count": len(values), "width": width, "logical_bits": logical_bits}


def unpack_indices(payload: dict[str, object], device: torch.device | str = "cpu") -> Tensor:
    """Decode indices stored by :func:`pack_indices`."""

    data = payload["data"]
    if not isinstance(data, Tensor):
        raise TypeError("packed index data must be a tensor")
    rank = int.from_bytes(bytes(data.cpu().tolist()), byteorder="little")
    count = int(payload["count"])
    width = int(payload["width"])
    values = [0] * count
    upper = width - 1
    for order in range(count, 0, -1):
        value = upper
        while math.comb(value, order) > rank:
            value -= 1
        values[order - 1] = value
        rank -= math.comb(value, order)
        upper = value - 1
    return torch.tensor(values, dtype=torch.long, device=device)


def _tensor_bits(tensor: Tensor) -> int:
    """Return physical dense tensor storage bits."""

    return tensor.numel() * tensor.element_size() * 8


def artifact_bpw_breakdown(
    artifacts: dict[str, object],
    refinement: dict[str, object] | None = None,
) -> dict[str, int | float]:
    """Audit logical payload and physical tensor storage against quantized weights."""

    totals = {
        "binary_signs_bits": 0, "masks_bits": 0, "scales_bits": 0,
        "thresholds_bits": 0, "means_zero_points_bits": 0, "low_rank_bits": 0,
        "rotation_bits": 0, "refinement_bits": 0, "padding_bits": 0,
        "activation_scales_bits": 0, "activation_zero_points_bits": 0,
        "metadata_bits": 0,
    }
    num_weights = 0
    for artifact_value in artifacts.values():
        artifact = artifact_value
        rows, columns = artifact["shape"]
        num_weights += int(rows) * int(columns)
        totals["metadata_bits"] += 328
        rotation_matrix = artifact.get("rotation_matrix")
        if isinstance(rotation_matrix, Tensor):
            totals["rotation_bits"] += _tensor_bits(rotation_matrix)
        for tensor in artifact["low_rank"].values():
            if isinstance(tensor, Tensor):
                totals["low_rank_bits"] += _tensor_bits(tensor)
        activation = artifact["activation"]
        if isinstance(activation.get("scales"), Tensor):
            totals["activation_scales_bits"] += _tensor_bits(activation["scales"])
        if isinstance(activation.get("zero_points"), Tensor):
            totals["activation_zero_points_bits"] += _tensor_bits(
                activation["zero_points"]
            )
        for block in artifact["blocks"]:
            totals["metadata_bits"] += 1408
            for key in ("salient_signs", "concentrated_signs", "sparse_signs"):
                payload = block[key]
                logical = int(payload["numel"])
                physical = _tensor_bits(payload["data"])
                totals["binary_signs_bits"] += logical
                totals["padding_bits"] += physical - logical
            salient_indices = block["salient_indices"]
            if isinstance(salient_indices, Tensor):
                totals["masks_bits"] += _tensor_bits(salient_indices)
            else:
                logical = int(salient_indices["logical_bits"])
                physical = _tensor_bits(salient_indices["data"])
                totals["masks_bits"] += logical
                totals["padding_bits"] += physical - logical
            payload = block["sparse_mask"]
            logical = int(payload["numel"])
            physical = _tensor_bits(payload["data"])
            totals["masks_bits"] += logical
            totals["padding_bits"] += physical - logical
            for key in ("salient_scales", "concentrated_scales", "sparse_scales"):
                totals["scales_bits"] += _tensor_bits(block[key])
            for key in ("salient_means", "concentrated_means", "sparse_means"):
                if isinstance(block.get(key), Tensor):
                    totals["means_zero_points_bits"] += _tensor_bits(block[key])
            if isinstance(block.get("thresholds"), Tensor):
                totals["thresholds_bits"] += _tensor_bits(block["thresholds"])
    if refinement is not None:
        for payload in refinement.values():
            for value in payload.values():
                if isinstance(value, Tensor):
                    totals["refinement_bits"] += _tensor_bits(value)
    if num_weights == 0:
        raise ValueError("no quantized weights found in artifacts")
    total_bits = sum(totals.values())
    result: dict[str, int | float] = {"num_weights": num_weights, **totals}
    for key, value in totals.items():
        result[key.replace("_bits", "_bpw")] = value / num_weights
    result["strict_binary_bpw"] = totals["binary_signs_bits"] / num_weights
    result["parameter_bpw"] = (
        totals["binary_signs_bits"] + totals["low_rank_bits"]
    ) / num_weights
    result["total_bits"] = total_bits
    result["total_bpw"] = total_bits / num_weights
    result["total_effective_bpw"] = total_bits / num_weights
    return result



def account_saved_artifact_storage(
    breakdown: dict[str, int | float],
    artifact_dir: Path | str,
    include_refinement: bool = False,
) -> dict[str, int | float]:
    """Replace estimated metadata with measured artifact-container overhead."""

    directory = Path(artifact_dir)
    names = ["binary_artifacts.pt", "low_rank.pt", "rotations.pt"]
    if include_refinement:
        names.append("refinement.pt")
    paths = [directory / name for name in names]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing artifact files for BPW accounting: {missing}")
    payload_keys = (
        "binary_signs_bits", "masks_bits", "scales_bits",
        "thresholds_bits", "means_zero_points_bits", "low_rank_bits",
        "rotation_bits", "refinement_bits", "padding_bits",
        "activation_scales_bits", "activation_zero_points_bits",
    )
    payload_bits = sum(int(breakdown[key]) for key in payload_keys)
    container_bits = sum(path.stat().st_size * 8 for path in paths)
    if container_bits < payload_bits:
        raise ValueError("artifact files are smaller than their tensor payload")
    result = dict(breakdown)
    result["metadata_bits"] = container_bits - payload_bits
    num_weights = int(result["num_weights"])
    for key in (*payload_keys, "metadata_bits"):
        result[key.replace("_bits", "_bpw")] = int(result[key]) / num_weights
    result["total_bits"] = container_bits
    result["total_bpw"] = container_bits / num_weights
    result["total_effective_bpw"] = container_bits / num_weights
    return result

import math

import torch

from billmv2.utils.artifacts import save_run_artifacts
from billmv2.utils.bits import (
    account_saved_artifact_storage,
    artifact_bpw_breakdown,
)


def _packed(shape: tuple[int, ...]) -> dict[str, object]:
    numel = math.prod(shape)
    return {
        "data": torch.zeros(math.ceil(numel / 8), dtype=torch.uint8),
        "shape": list(shape),
        "numel": numel,
        "true_count": 0,
    }


def _block(rows: int, width: int, salient_columns: int) -> dict[str, object]:
    salient = rows * salient_columns
    nonsalient = rows * (width - salient_columns)
    return {
        "start": 0,
        "end": width,
        "salient_indices": torch.arange(salient_columns, dtype=torch.uint8),
        "sparse_mask": _packed((rows, width)),
        "salient_signs": _packed((2, salient)),
        "concentrated_signs": _packed((1, nonsalient // 2)),
        "sparse_signs": _packed((1, nonsalient - nonsalient // 2)),
        "salient_scales": torch.zeros(rows, 2, dtype=torch.float16),
        "concentrated_scales": torch.zeros(rows, 1, dtype=torch.float16),
        "sparse_scales": torch.zeros(rows, 1, dtype=torch.float16),
        "thresholds": torch.zeros(2, dtype=torch.float16),
    }


def test_manual_small_artifact_breakdown() -> None:
    block = _block(2, 4, 1)
    artifacts = {
        "linear": {
            "shape": [2, 4],
            "rotation_matrix": None,
            "low_rank": {},
            "activation": {
                "bits": 16, "group_size": 128, "symmetric": True,
                "clip_method": "mse", "feature_dim": 4,
            },
            "blocks": [block],
        }
    }
    result = artifact_bpw_breakdown(artifacts)
    assert result["num_weights"] == 8
    assert result["binary_signs_bits"] == 10
    assert result["masks_bits"] == 16
    assert result["scales_bits"] == 128
    assert result["thresholds_bits"] == 32
    assert result["padding_bits"] == 14


def test_4096_rank4_fp16_low_rank_overhead() -> None:
    rows = columns = 4096
    blocks = []
    for start in range(0, columns, 128):
        block = _block(rows, 128, 13)
        block["start"] = start
        block["end"] = start + 128
        blocks.append(block)
    low_rank = {
        "u": torch.zeros(rows, 4, dtype=torch.float16),
        "v": torch.zeros(columns, 4, dtype=torch.float16),
    }
    artifacts = {
        "linear": {
            "shape": [rows, columns],
            "rotation_matrix": None,
            "low_rank": low_rank,
            "activation": {
                "bits": 16, "group_size": 128, "symmetric": True,
                "clip_method": "mse", "feature_dim": columns,
            },
            "blocks": blocks,
        }
    }
    result = artifact_bpw_breakdown(artifacts)
    assert low_rank["u"].dtype == low_rank["v"].dtype == torch.float16
    assert low_rank["u"].shape == low_rank["v"].shape == (4096, 4)
    assert result["low_rank_bits"] == 16 * 4 * (4096 + 4096)
    assert result["low_rank_bpw"] == 0.03125


def test_saved_tensor_layout_matches_accounting(tmp_path) -> None:
    artifacts = {
        "linear": {
            "shape": [2, 4],
            "rotation_matrix": None,
            "low_rank": {
                "u": torch.zeros(2, 1, dtype=torch.float16),
                "v": torch.zeros(4, 1, dtype=torch.float16),
            },
            "activation": {
                "bits": 4, "group_size": 4, "symmetric": True,
                "clip_method": "mse", "feature_dim": 4,
                "scales": torch.ones(1, dtype=torch.float16),
            },
            "blocks": [_block(2, 4, 1)],
        }
    }
    breakdown = artifact_bpw_breakdown(artifacts)
    save_run_artifacts(tmp_path, {"model": "tiny"}, artifacts, {}, breakdown, {})
    binary = torch.load(tmp_path / "binary_artifacts.pt")
    low_rank = torch.load(tmp_path / "low_rank.pt")
    block = binary["linear"]["blocks"][0]
    assert block["salient_indices"].dtype == torch.uint8
    assert "concentrated_mask" not in block
    assert low_rank["linear"]["u"].dtype == torch.float16
    assert low_rank["linear"]["u"].shape == (2, 1)
    assert low_rank["linear"]["v"].shape == (4, 1)
    assert breakdown["low_rank_bits"] == 16 * (2 + 4)
    measured = account_saved_artifact_storage(breakdown, tmp_path)
    stored_bits = sum(
        (tmp_path / name).stat().st_size * 8
        for name in ("binary_artifacts.pt", "low_rank.pt", "rotations.pt")
    )
    assert measured["total_bits"] == stored_bits
    assert measured["metadata_bits"] == stored_bits - sum(
        measured[key]
        for key in measured
        if key.endswith("_bits") and key not in {"metadata_bits", "total_bits"}
    )


def test_low_rank_storage_is_accounted_for_independently() -> None:
    artifacts = {
        "linear": {
            "shape": [4096, 4096],
            "rotation_matrix": None,
            "low_rank": {},
            "activation": {
                "bits": 4, "group_size": 128, "symmetric": True,
                "clip_method": "mse", "feature_dim": 4096,
                "scales": torch.ones(32, dtype=torch.float16),
            },
            "blocks": [
                {**_block(4096, 128, 13), "start": start, "end": start + 128}
                for start in range(0, 4096, 128)
            ],
        }
    }
    rank0 = artifact_bpw_breakdown(artifacts)
    artifacts["linear"]["low_rank"] = {
        "u": torch.zeros(4096, 4, dtype=torch.float16),
        "v": torch.zeros(4096, 4, dtype=torch.float16),
    }
    rank4 = artifact_bpw_breakdown(artifacts)
    assert rank4["masks_bits"] == rank0["masks_bits"]
    assert rank4["scales_bits"] == rank0["scales_bits"]
    assert rank4["low_rank_bpw"] - rank0["low_rank_bpw"] == 0.03125

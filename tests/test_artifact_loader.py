import json
from pathlib import Path

import torch
from torch import nn

from billmv2.config import BiLLMv2Config
from billmv2.quantization.v2_quantizer import BiLLMv2Quantizer
from billmv2.utils.artifacts import (
    apply_billmv2_artifacts,
    load_billmv2_artifacts,
    save_run_artifacts,
)
from billmv2.utils.bits import artifact_bpw_breakdown
from billmv2.quantization.activation import install_activation_quantizer


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(8, 6, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


def _save_tiny(directory: Path) -> tuple[_TinyModel, torch.Tensor, torch.Tensor]:
    torch.manual_seed(13)
    model = _TinyModel()
    quantizer = BiLLMv2Quantizer(
        model.linear,
        BiLLMv2Config(
            model="tiny",
            nsamples=2,
            blocksize=4,
            low_rank_rank=2,
            split_candidates=4,
            split_rerank_topk=2,
        ),
    )
    quantizer.add_batch(torch.randn(2, 5, 8))
    result = quantizer.quantize()
    install_activation_quantizer(model.linear, result.artifact["activation"])
    inputs = torch.randn(3, 8)
    expected = model(inputs)
    artifacts = {"linear": result.artifact}
    save_run_artifacts(
        directory,
        {"model": "tiny", "seed": 0},
        artifacts,
        {},
        artifact_bpw_breakdown(artifacts),
        {},
    )
    return model, inputs, expected


def test_ptq_bundle_reload_matches_saved_output(tmp_path: Path) -> None:
    model, inputs, expected = _save_tiny(tmp_path)
    fresh = _TinyModel()
    bundle = load_billmv2_artifacts(tmp_path)
    apply_billmv2_artifacts(fresh, bundle)
    torch.testing.assert_close(fresh(inputs), expected, atol=2e-3, rtol=2e-3)
    assert bundle["refinement"] is None


def test_refinement_reload_and_unknown_version(tmp_path: Path) -> None:
    quantized, inputs, _ = _save_tiny(tmp_path)
    refinement = {
        "linear": {
            "scale": torch.full((6, 1), 1.01),
            "u": torch.zeros(6, 2),
            "v": torch.zeros(8, 2),
            "rotation_mode": "block_cayley",
        }
    }
    torch.save(refinement, tmp_path / "refinement.pt")
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config["artifact_type"] = "ptq_ft"
    config_path.write_text(json.dumps(config))
    fresh = _TinyModel()
    bundle = load_billmv2_artifacts(tmp_path)
    apply_billmv2_artifacts(fresh, bundle)
    torch.testing.assert_close(
        fresh(inputs), quantized(inputs) * 1.01, atol=2e-3, rtol=2e-3
    )
    torch.save(
        {"linear": {"scale": torch.ones(6, 1), "rotation_mode": "none"}},
        tmp_path / "refinement.pt",
    )
    selective = _TinyModel()
    apply_billmv2_artifacts(selective, load_billmv2_artifacts(tmp_path))
    torch.testing.assert_close(
        selective(inputs), quantized(inputs), atol=2e-3, rtol=2e-3
    )
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config["artifact_format_version"] = 999
    config_path.write_text(json.dumps(config))
    try:
        load_billmv2_artifacts(tmp_path)
    except ValueError as error:
        assert "unsupported artifact format version" in str(error)
    else:
        raise AssertionError("unknown artifact version must fail")

#!/usr/bin/env python3
"""Run pure BiLLM-v2 post-training quantization."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import torch

from billmv2.baseline import evaluate_perplexity, evaluate_perplexity_limited, load_model
from billmv2.calibration.data import get_candidate_pool
from billmv2.calibration.features import extract_calibration_features
from billmv2.calibration.selector import SelectionResult, select_calibration
from billmv2.config import BiLLMv2Config
from billmv2.pipeline.ptq import run_ptq_pipeline
from billmv2.utils.artifacts import apply_billmv2_artifacts, load_billmv2_artifacts, save_run_artifacts
from billmv2.utils.bits import artifact_bpw_breakdown
from billmv2.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


F2_PRESET_OVERRIDES: dict[str, Any] = {
    "nsamples": 128,
    "seqlen": 2048,
    "calib_candidate_size": 512,
    "calib_selector": "kcenter",
    "calib_feature": "activation",
    "calib_feature_dim": 64,
    "calib_probe_stride": 4,
    "seed": 0,
    "blocksize": 128,
    "percdamp": 0.01,
    "rotation": "none",
    "rotation_block_size": 128,
    "linear_basis_rotation": "hadamard",
    "vo_rotation_candidates": ["identity", "signed_hadamard", "random_orthogonal", "covariance_hadamard"],
    "rotation_candidate_seeds": [0, 1],
    "rotation_fit_samples": 32,
    "rotation_validation_samples": 16,
    "low_rank_rank": 0,
    "low_rank_mode": "functional_branch",
    "low_rank_metric": "diag_hessian",
    "low_rank_dtype": "int8",
    "low_rank_int8_scale_mode": "tensor",
    "functional_low_rank_ranks": [0, 2, 4],
    "functional_low_rank_max_rank": 4,
    "functional_lr_alternating_steps": 1,
    "functional_fit_samples": 0,
    "functional_validation_samples": 0,
    "functional_candidate_topk": 2,
    "functional_lookahead_margin": 0.005,
    "functional_lookahead_blocks": 1,
    "global_functional_budget_topup": False,
    "target_parameter_bpw": 1.1015625,
    "topup_objective": "block_validation_gain_per_bit",
    "selective_vo_rotation": False,
    "max_rotation_rescue_blocks": 8,
    "rotation_acceptance_margin": 0.01,
    "low_rank_ridge": 0.0001,
    "low_rank_sketch_dim": 128,
    "low_rank_max_tokens": 4096,
    "joint_rotation_low_rank_search": True,
    "joint_search_coarse_samples": 8,
    "joint_search_final_samples": 16,
    "joint_search_topk": 2,
    "fixed_bpw_low_rank": False,
    "fixed_bpw_target": 1.101563,
    "activation_bits": 16,
    "activation_group_size": 128,
    "activation_symmetric": True,
    "activation_clip_method": "mse",
    "salient_metric": "residual_hessian",
    "salient_fraction": 0.1,
    "coupled_saliency": "independent_residual_hessian",
    "saliency_l2_lambdas": [1.0],
    "split_mode": "asymmetric",
    "split_granularity": "global",
    "split_candidates": 16,
    "split_rerank_topk": 4,
    "row_split_rerank": "none",
    "split_row_tile": 256,
    "alternating_steps": 2,
    "geometry_loss": "diagonal_whiten",
    "geometry_gamma": 0.5,
    "geometry_eps": 1e-5,
    "minlayer": -1,
    "maxlayer": 1000,
    "max_layers": 0,
    "target_modules": [],
    "quant_only": "",
    "invert": False,
    "disable_gptq": False,
}


def build_parser() -> argparse.ArgumentParser:
    """Build the shared PTQ command-line parser."""

    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("calib_dataset", choices=["c4", "wikitext2", "mixed"])
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=0)
    parser.add_argument("--preset", choices=["billmv2_flr_f2"], default=None)
    parser.add_argument("--calib_candidate_size", type=int, default=1024)
    parser.add_argument(
        "--calib_feature", choices=["proxy", "activation", "weight_error", "activation_error", "joint"], default="joint"
    )
    parser.add_argument("--calib_feature_dim", type=int, default=64)
    parser.add_argument("--calib_probe_stride", type=int, default=4)
    parser.add_argument("--calib_feature_cache", default="")
    parser.add_argument(
        "--calib_selector",
        choices=[
            "random", "kcenter", "d_optimal", "hybrid",
            "kcenter_activation", "d_optimal_joint", "hybrid_joint",
        ],
        default="hybrid",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument(
        "--rotation",
        choices=["none", "linear_basis", "vo_foldable", "hadamard", "random_orthogonal"],
        default="none",
    )
    parser.add_argument("--rotation_block_size", type=int, default=128)
    parser.add_argument(
        "--linear_basis_rotation", choices=["hadamard", "random_orthogonal"],
        default="hadamard",
    )
    parser.add_argument(
        "--vo_rotation_candidates", nargs="+",
        choices=["identity", "signed_hadamard", "random_orthogonal", "covariance_hadamard"],
        default=["identity", "signed_hadamard", "random_orthogonal", "covariance_hadamard"],
    )
    parser.add_argument("--rotation_candidate_seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--rotation_fit_samples", type=int, default=32)
    parser.add_argument("--rotation_validation_samples", type=int, default=16)
    parser.add_argument("--low_rank_rank", type=int, choices=[0, 2, 4, 8, 16], default=0)
    parser.add_argument(
        "--low_rank_mode", choices=["none", "weight_residual", "functional_branch"],
        default="none",
    )
    parser.add_argument(
        "--low_rank_metric",
        choices=["weight", "diag_hessian", "full_hessian"],
        default="diag_hessian",
    )
    parser.add_argument("--low_rank_dtype", choices=["fp16", "int8"], default="fp16")
    parser.add_argument("--low_rank_int8_scale_mode", choices=["tensor", "per_rank"], default="tensor")
    parser.add_argument(
        "--functional_low_rank_ranks", nargs="+", type=int,
        choices=[0, 2, 4, 6, 8, 12], default=[0, 2, 4],
    )
    parser.add_argument("--functional_low_rank_max_rank", type=int, choices=[0, 2, 4, 6, 8, 12], default=4)
    parser.add_argument("--functional_lr_alternating_steps", type=int, choices=[1, 2], default=1)
    parser.add_argument("--functional_fit_samples", type=int, default=0)
    parser.add_argument("--functional_validation_samples", type=int, default=0)
    parser.add_argument("--functional_candidate_topk", type=int, default=2)
    parser.add_argument("--functional_lookahead_margin", type=float, default=0.005)
    parser.add_argument("--functional_lookahead_blocks", type=int, default=1)
    parser.add_argument("--global_functional_budget_topup", action="store_true")
    parser.add_argument("--target_parameter_bpw", type=float, default=1.1015625)
    parser.add_argument(
        "--topup_objective", choices=["block_validation_gain_per_bit"],
        default="block_validation_gain_per_bit",
    )
    parser.add_argument("--selective_vo_rotation", action="store_true")
    parser.add_argument("--max_rotation_rescue_blocks", type=int, default=8)
    parser.add_argument("--rotation_acceptance_margin", type=float, default=0.01)
    parser.add_argument("--low_rank_ridge", type=float, default=1e-4)
    parser.add_argument("--low_rank_sketch_dim", type=int, default=128)
    parser.add_argument("--low_rank_max_tokens", type=int, default=4096)
    parser.add_argument("--joint_rotation_low_rank_search", action="store_true")
    parser.add_argument("--joint_search_coarse_samples", type=int, default=8)
    parser.add_argument("--joint_search_final_samples", type=int, default=16)
    parser.add_argument("--joint_search_topk", type=int, default=2)
    parser.add_argument("--fixed_bpw_low_rank", action="store_true")
    parser.add_argument("--fixed_bpw_target", type=float, default=1.101563)
    parser.add_argument("--activation_bits", type=int, choices=[4, 8, 16], default=4)
    parser.add_argument("--activation_group_size", type=int, default=128)
    parser.add_argument(
        "--activation_symmetric", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--activation_clip_method", choices=["max", "percentile", "mse"], default="mse"
    )
    parser.add_argument(
        "--salient_metric",
        choices=["magnitude", "hessian", "residual_hessian"],
        default="residual_hessian",
    )
    parser.add_argument("--salient_fraction", type=float, default=0.1)
    parser.add_argument(
        "--coupled_saliency",
        choices=[
            "independent_residual_hessian", "coupled_qkv", "coupled_gate_up",
            "coupled_qkv_gate_up",
        ],
        default="independent_residual_hessian",
    )
    parser.add_argument("--saliency_l2_lambdas", nargs="+", type=float, default=[1.0])
    parser.add_argument(
        "--split_mode",
        choices=["original", "symmetric", "asymmetric", "family_rerank"],
        default="asymmetric",
    )
    parser.add_argument("--split_granularity", choices=["global", "per_row"], default="global")
    parser.add_argument("--split_candidates", type=int, default=16)
    parser.add_argument("--split_rerank_topk", type=int, default=4)
    parser.add_argument("--row_split_rerank", choices=["none", "linear_top2"], default="none")
    parser.add_argument("--split_row_tile", type=int, default=256)
    parser.add_argument("--alternating_steps", type=int, default=1)
    parser.add_argument(
        "--geometry_loss",
        choices=["none", "mse", "diagonal_whiten"],
        default="diagonal_whiten",
    )
    parser.add_argument("--geometry_gamma", type=float, default=0.5)
    parser.add_argument("--geometry_eps", type=float, default=1e-5)
    parser.add_argument("--minlayer", type=int, default=-1)
    parser.add_argument("--maxlayer", type=int, default=1000)
    parser.add_argument("--max_layers", type=int, default=0)
    parser.add_argument("--target_modules", nargs="+", default=[])
    parser.add_argument("--quant_only", default="")
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--disable_gptq", action="store_true")
    parser.add_argument("--eval_dataset", choices=["none", "wikitext2", "c4", "ptb"], default="wikitext2")
    parser.add_argument("--eval_max_samples", type=int, default=0)
    parser.add_argument("--output_dir", default="/autodl-fs/data/cclanro/billm-v2-output/default")
    parser.add_argument("--save_merged_model", action="store_true")
    parser.add_argument("--validate_reload", action="store_true")
    parser.add_argument("--calib_synthesize", action="store_true")
    parser.add_argument("--synthesis_num_samples", type=int, default=32)
    return parser


def apply_preset(args: argparse.Namespace) -> None:
    """Apply explicit method presets."""

    if args.preset == "billmv2_flr_f2":
        for name, value in F2_PRESET_OVERRIDES.items():
            setattr(args, name, value)
        return
    selector_aliases = {
        "kcenter_activation": ("kcenter", "activation"),
        "d_optimal_joint": ("d_optimal", "joint"),
        "hybrid_joint": ("hybrid", "joint"),
    }
    if args.calib_selector in selector_aliases:
        args.calib_selector, args.calib_feature = selector_aliases[args.calib_selector]


def config_from_args(args: argparse.Namespace) -> BiLLMv2Config:
    """Create the immutable core configuration from parsed arguments."""

    fields = BiLLMv2Config.__dataclass_fields__
    values = {name: getattr(args, name) for name in fields if hasattr(args, name)}
    values["target_modules"] = tuple(values.get("target_modules", ()))
    values["saliency_l2_lambdas"] = tuple(values.get("saliency_l2_lambdas", (1.0,)))
    values["vo_rotation_candidates"] = tuple(values.get("vo_rotation_candidates", ()))
    values["rotation_candidate_seeds"] = tuple(values.get("rotation_candidate_seeds", (0, 1)))
    values["functional_low_rank_ranks"] = tuple(values.get("functional_low_rank_ranks", (0, 2, 4)))
    if values.get("functional_low_rank_max_rank", 4) < max(values["functional_low_rank_ranks"], default=0):
        values["functional_low_rank_max_rank"] = max(values["functional_low_rank_ranks"], default=0)
    return BiLLMv2Config(**values)


def _candidate_features(samples: list[Any], dimension: int = 32) -> torch.Tensor:
    features = torch.zeros((len(samples), dimension), dtype=torch.float32)
    for index, sample in enumerate(samples):
        tokens = (sample[0] if isinstance(sample, (tuple, list)) else sample).reshape(-1).long()
        features[index].scatter_add_(
            0, torch.remainder(tokens, dimension), torch.ones_like(tokens, dtype=torch.float32)
        )
        features[index] /= features[index].norm().clamp_min(1.0)
    return features



def run(args: argparse.Namespace) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Execute PTQ and persist compact artifacts."""

    apply_preset(args)
    config = config_from_args(args)
    output_dir = Path(config.output_dir)
    configure_logging(output_dir / "run.log")
    LOGGER.info("PTQ mode: no backward, no optimizer")
    model = load_model(config.model).eval()
    if config.seqlen > 0:
        model.seqlen = config.seqlen
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("pure PTQ requires every original parameter to be frozen")
    if torch.cuda.is_available():
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(config.device)
    quantization_start = time.perf_counter()
    candidates, test_loader = get_candidate_pool(
        config.calib_dataset,
        max(config.nsamples, config.calib_candidate_size),
        config.seed,
        model.seqlen,
        config.model,
    )
    if config.calib_feature == "proxy":
        features = _candidate_features(candidates, config.calib_feature_dim)
    else:
        feature_cache = (
            Path(args.calib_feature_cache)
            if args.calib_feature_cache
            else output_dir / "calibration_features.pt"
        )
        cache_key = {
            "model": config.model, "dataset": config.calib_dataset,
            "candidate_size": len(candidates), "seed": config.seed,
            "seqlen": model.seqlen, "feature": config.calib_feature,
            "feature_dim": config.calib_feature_dim,
            "probe_stride": config.calib_probe_stride,
            "activation_bits": config.activation_bits,
            "activation_group_size": config.activation_group_size,
            "activation_symmetric": config.activation_symmetric,
            "activation_clip_method": config.activation_clip_method,
        }
        cached = torch.load(feature_cache) if feature_cache.exists() else None
        if isinstance(cached, dict) and cached.get("key") == cache_key:
            features = cached["features"]
        else:
            features = extract_calibration_features(
                model, candidates, config.device, config.calib_feature_dim,
                config.calib_probe_stride, config.calib_feature, config.seed,
                activation_bits=config.activation_bits,
                activation_group_size=config.activation_group_size,
                activation_symmetric=config.activation_symmetric,
                activation_clip_method=config.activation_clip_method,
                rotation=config.rotation,
                rotation_block_size=config.rotation_block_size,
            )
            feature_cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"key": cache_key, "features": features}, feature_cache)
    selection: SelectionResult = select_calibration(
        features,
        config.nsamples,
        config.calib_selector,
        config.seed,
    )
    calibration_loader = [candidates[index] for index in selection.indices]
    if getattr(args, "capture_fp_targets", False):
        fp_logits = []
        fp_hidden = []
        model.to(config.device)
        with torch.no_grad():
            for sample in calibration_loader:
                token_ids = sample[0] if isinstance(sample, (tuple, list)) else sample
                outputs = model(token_ids.to(config.device), output_hidden_states=True, use_cache=False)
                fp_logits.append(outputs.logits[:, -1].half().cpu())
                fp_hidden.append(outputs.hidden_states[-1][:, -1].half().cpu())
        model.cpu()
        model._billmv2_fp_logits = fp_logits
        model._billmv2_fp_hidden = fp_hidden
        model._billmv2_calibration_loader = calibration_loader
        torch.cuda.empty_cache()
    artifacts, geometry = run_ptq_pipeline(model, calibration_loader, config)
    quantization_time_s = time.perf_counter() - quantization_start
    peak_gpu_memory_gib = (
        torch.cuda.max_memory_allocated(config.device) / 1024**3
        if torch.cuda.is_available()
        else 0.0
    )
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("pure PTQ unexpectedly enabled gradients")
    calibration = {
        "indices": selection.indices,
        "marginal_scores": selection.marginal_scores,
        "information_scores": selection.information_scores,
    }
    metrics: dict[str, Any] = {
        "geometry_by_layer": geometry,
        "quantization_time_s": quantization_time_s,
        "peak_gpu_memory_gib": peak_gpu_memory_gib,
    }
    low_rank_branches = 0
    low_rank_macs_per_token = 0
    original_linear_macs_per_token = 0
    for artifact in artifacts.values():
        shape = artifact.get("shape", ())
        rows = 0
        columns = 0
        if len(shape) == 2:
            rows, columns = int(shape[0]), int(shape[1])
            original_linear_macs_per_token += rows * columns
        low_rank = artifact.get("low_rank")
        if isinstance(low_rank, dict) and int(low_rank.get("rank", 0)) > 0:
            rank = int(low_rank["rank"])
            low_rank_branches += 1
            low_rank_macs_per_token += rank * (rows + columns)
    metrics.update(
        {
            "selected_low_rank_branches": low_rank_branches,
            "total_low_rank_macs_per_token": low_rank_macs_per_token,
            "low_rank_macs_over_original_linear_macs": (
                low_rank_macs_per_token / original_linear_macs_per_token
                if original_linear_macs_per_token > 0 else 0.0
            ),
            "rotation_search_time_s": sum(
                float(value) for key, value in geometry.items()
                if key.endswith("_rotation_search_time_s")
            ),
            "functional_low_rank_fit_time_s": sum(
                float(value) for key, value in geometry.items()
                if key.endswith("_functional_low_rank_time_s")
            ),
            "joint_search_time_s": sum(
                float(value) for key, value in geometry.items()
                if key.endswith("_joint_search_time_s")
            ),
        }
    )
    bpw = artifact_bpw_breakdown(artifacts)
    if config.joint_rotation_low_rank_search:
        if bpw["parameter_bpw"] > 1.1015625 + 1e-9:
            raise AssertionError(f"joint search exceeded parameter BPW: {bpw['parameter_bpw']}")
        if bpw["masks_bpw"] > 1.000123 + 1e-9:
            raise AssertionError(f"joint search exceeded mask BPW: {bpw['masks_bpw']}")
        if bpw["scales_bpw"] > 0.500000 + 1e-9:
            raise AssertionError(f"joint search exceeded scale BPW: {bpw['scales_bpw']}")
    save_run_artifacts(
        output_dir, config.to_dict(), artifacts, metrics,
        bpw, calibration,
    )
    if args.eval_dataset != "none":
        if args.eval_dataset != config.calib_dataset or test_loader is None:
            _, test_loader = get_candidate_pool(
                args.eval_dataset, 1, config.seed, model.seqlen, config.model
            )
        metrics["ppl"] = (
            evaluate_perplexity_limited(model, test_loader, config.device, args.eval_max_samples)
            if args.eval_max_samples > 0
            else evaluate_perplexity(model, test_loader, config.device, args.eval_dataset)
        )
    (output_dir / "metrics.json").write_text(
        __import__("json").dumps(metrics, indent=2), encoding="utf-8"
    )
    if args.validate_reload:
        validation_tokens = calibration_loader[0][0][:, : min(model.seqlen, 128)]
        model.to(config.device).eval()
        with torch.no_grad():
            reference = model(validation_tokens.to(config.device), use_cache=False).logits.cpu()
        model.cpu()
        reloaded = load_model(config.model).eval()
        if config.seqlen > 0:
            reloaded.seqlen = config.seqlen
        bundle = load_billmv2_artifacts(output_dir)
        apply_billmv2_artifacts(reloaded, bundle)
        reloaded.to(config.device)
        with torch.no_grad():
            restored = reloaded(validation_tokens.to(config.device), use_cache=False).logits.cpu()
        reloaded.cpu()
        reload_error = float((reference.float() - restored.float()).abs().max())
        if not torch.isfinite(torch.tensor(reload_error)):
            raise FloatingPointError("compact reload output error is NaN or Inf")
        metrics["compact_reload_max_abs_error"] = reload_error
        (output_dir / "metrics.json").write_text(
            __import__("json").dumps(metrics, indent=2), encoding="utf-8"
        )
        LOGGER.info("compact reload max output error: %.9g", reload_error)
    if config.save_merged_model:
        model.save_pretrained(output_dir / "merged_model")
    LOGGER.info("artifacts written to %s", output_dir)
    return model, artifacts


if __name__ == "__main__":
    run(build_parser().parse_args())

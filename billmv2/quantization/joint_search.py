"""Local fixed-budget V/O rotation and functional branch search."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import logging
import time
from typing import Any, Callable

import torch
from torch import Tensor, nn

from billmv2.config import BiLLMv2Config
from billmv2.low_rank.functional import (
    fit_functional_spectrum,
    functional_energy_explained,
    functional_factor_payload,
    functional_parameter_bits,
    install_functional_adapter,
    remove_functional_adapter,
    truncate_functional_spectrum,
)
from billmv2.reconstruction.objectives import geometry_loss, geometry_weights
from billmv2.transforms.vo_rotation import (
    VoRotation,
    fold_vo_weights,
    make_vo_rotation_candidates,
)

from .activation import install_activation_quantizer
from .v2_quantizer import BiLLMv2Quantizer, QuantizationResult

LOGGER = logging.getLogger(__name__)
RunLayer = Callable[[nn.Module, Tensor, dict[str, Any]], Tensor]


@dataclass
class BranchTargets:
    """Hold branch outputs before residual additions for search samples."""

    attention: Tensor
    mlp: Tensor


@dataclass
class BranchCandidate:
    """Store one serially evaluated branch candidate on CPU."""

    name: str
    loss: float
    parameter_bits: int
    salient_columns: int
    rank: int
    primary_weight: Tensor
    primary_artifact: dict[str, Any]
    secondary_weight: Tensor | None = None
    secondary_artifact: dict[str, Any] | None = None
    low_rank_payload: dict[str, Any] | None = None
    rotation: str = "identity"


@dataclass(frozen=True)
class JointSearchResult:
    """Return selected V/O/down results and search accounting."""

    results: dict[str, QuantizationResult]
    statistics: dict[str, float | str]


def capture_teacher_branches(
    layer: nn.Module,
    inputs: Tensor,
    kwargs: dict[str, Any],
    run_layer: RunLayer,
    sample_limit: int,
) -> tuple[Tensor, BranchTargets]:
    """Capture attention/MLP outputs before their residual additions."""

    attention_rows: list[Tensor] = []
    mlp_rows: list[Tensor] = []

    def attention_hook(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> None:
        if len(attention_rows) < sample_limit:
            value = output[0] if isinstance(output, tuple) else output
            attention_rows.append(value.detach().half().cpu())

    def mlp_hook(_module: nn.Module, _args: tuple[Any, ...], output: Tensor) -> None:
        if len(mlp_rows) < sample_limit:
            mlp_rows.append(output.detach().half().cpu())

    handles = [
        layer.self_attn.register_forward_hook(attention_hook),
        layer.mlp.register_forward_hook(mlp_hook),
    ]
    try:
        outputs = run_layer(layer, inputs, kwargs)
    finally:
        for handle in handles:
            handle.remove()
    return outputs, BranchTargets(torch.cat(attention_rows), torch.cat(mlp_rows))


def _salient_sign_bits(rows: int, columns: int, block_size: int, count: int) -> int:
    selected = sum(min(count, columns - start) for start in range(0, columns, block_size))
    return rows * columns + rows * selected


def local_candidate_parameter_bits(
    module: nn.Linear,
    block_size: int,
    salient_columns: int,
    rank: int,
    factor_dtype: str,
    int8_scale_mode: str = "tensor",
) -> int:
    """Count active binary signs and functional factor payload bits."""

    return _salient_sign_bits(
        module.out_features, module.in_features, block_size, salient_columns
    ) + functional_parameter_bits(
        module.in_features, module.out_features, rank, factor_dtype, int8_scale_mode
    )


def _candidate_quantizer(
    source: BiLLMv2Quantizer,
    module: nn.Linear,
    weight: Tensor,
    salient_columns: int,
) -> QuantizationResult:
    config = replace(
        source.config,
        rotation="none",
        low_rank_mode="none",
        low_rank_rank=0,
        fixed_bpw_low_rank=False,
        salient_fraction=salient_columns / source.config.blocksize,
    )
    module.weight.data.copy_(weight.to(module.weight))
    quantizer = BiLLMv2Quantizer(module, config, source.module_name)
    quantizer.hessian.copy_(source.hessian)
    quantizer.num_tokens = source.num_tokens
    quantizer.num_samples = source.num_samples
    quantizer.inputs = list(source.inputs)
    quantizer.activation = deepcopy(source.activation)
    return quantizer.quantize()


def _collect_quantizer(
    layer: nn.Module,
    module: nn.Linear,
    module_name: str,
    inputs: Tensor,
    kwargs: dict[str, Any],
    run_layer: RunLayer,
    config: BiLLMv2Config,
) -> BiLLMv2Quantizer:
    quantizer = BiLLMv2Quantizer(module, replace(config, rotation="none"), module_name)

    def collect(_module: nn.Module, arguments: tuple[Tensor, ...]) -> None:
        quantizer.add_batch(arguments[0])

    handle = module.register_forward_pre_hook(collect)
    try:
        run_layer(layer, inputs, kwargs)
    finally:
        handle.remove()
    quantizer.finalize_activation()
    return quantizer


def _capture_branch(
    layer: nn.Module,
    module: nn.Linear,
    branch: nn.Module,
    inputs: Tensor,
    kwargs: dict[str, Any],
    run_layer: RunLayer,
) -> tuple[Tensor, Tensor, Tensor]:
    module_inputs: list[Tensor] = []
    branch_outputs: list[Tensor] = []

    def input_hook(_module: nn.Module, arguments: tuple[Tensor, ...]) -> None:
        module_inputs.append(arguments[0].detach())

    def output_hook(_module: nn.Module, _arguments: tuple[Any, ...], output: Any) -> None:
        value = output[0] if isinstance(output, tuple) else output
        branch_outputs.append(value.detach())

    handles = [module.register_forward_pre_hook(input_hook), branch.register_forward_hook(output_hook)]
    try:
        block_outputs = run_layer(layer, inputs, kwargs)
    finally:
        for handle in handles:
            handle.remove()
    return torch.cat(module_inputs), torch.cat(branch_outputs), block_outputs


def _fit_rank_payloads(
    z: Tensor,
    teacher: Tensor,
    rank0: Tensor,
    ranks: tuple[int, ...],
    config: BiLLMv2Config,
    target_name: str,
) -> dict[int, dict[str, Any]]:
    max_rank = max(config.functional_low_rank_max_rank, max(ranks, default=0))
    spectrum = fit_functional_spectrum(
        z, teacher.to(z.device) - rank0, max_rank,
        config.low_rank_ridge, config.low_rank_sketch_dim,
        config.low_rank_max_tokens, config.seed,
    )
    payloads: dict[int, dict[str, Any]] = {}
    for rank in ranks:
        if rank > spectrum.max_rank:
            continue
        factors = truncate_functional_spectrum(spectrum, rank, z)
        metadata = {
            "fit_rank_max": int(spectrum.max_rank),
            "fit_rank_selected": int(rank),
            "functional_energy_explained": functional_energy_explained(spectrum, rank),
            "singular_values": [float(value) for value in spectrum.singular.detach().cpu()],
        }
        payload, _, _ = functional_factor_payload(
            factors, config.low_rank_dtype, target_name,
            config.low_rank_int8_scale_mode, metadata,
        )
        payloads[rank] = payload
    return payloads


def _branch_loss(student: Tensor, teacher: Tensor, config: BiLLMv2Config) -> float:
    target = teacher.to(student.device)
    weights = geometry_weights(target, config.geometry_loss, config.geometry_gamma, config.geometry_eps)
    return float(geometry_loss(student, target, weights))


def _forced_topk(
    candidates: list[BranchCandidate], baseline_name: str, topk: int
) -> list[BranchCandidate]:
    ordered = sorted(candidates, key=lambda item: item.loss)
    selected = ordered[:topk]
    baseline = next(item for item in candidates if item.name == baseline_name)
    if baseline not in selected:
        selected[-1] = baseline
        selected.sort(key=lambda item: item.loss)
    return selected


@torch.no_grad()
def search_block_rotation_low_rank(
    layer: nn.Module,
    quantizers: dict[str, BiLLMv2Quantizer],
    student_inputs: Tensor,
    teacher_outputs: Tensor,
    teacher_targets: BranchTargets,
    kwargs: dict[str, Any],
    run_layer: RunLayer,
    config: BiLLMv2Config,
) -> JointSearchResult:
    """Run serial local V/O/down search under the baseline sign budget."""

    started = time.perf_counter()
    required = ("self_attn.v_proj", "self_attn.o_proj", "mlp.down_proj")
    if any(name not in quantizers for name in required):
        raise KeyError("joint search requires LLaMA v_proj, o_proj, and down_proj")
    v_module = layer.self_attn.v_proj
    o_module = layer.self_attn.o_proj
    down_module = layer.mlp.down_proj
    original_v = v_module.weight.detach().clone()
    original_o = o_module.weight.detach().clone()
    original_down = down_module.weight.detach().clone()
    requested_fit = config.functional_fit_samples or config.rotation_fit_samples
    requested_validation = config.functional_validation_samples or config.rotation_validation_samples
    sample_count = min(
        student_inputs.shape[0],
        max(requested_fit + requested_validation, config.joint_search_final_samples),
    )
    fit_count = min(requested_fit, max(1, sample_count // 2))
    validation_start = fit_count
    validation_count = min(requested_validation, sample_count - validation_start)
    if validation_count == 0:
        validation_start, validation_count = 0, sample_count
    if config.functional_validation_samples:
        coarse_count = min(16, validation_count)
        final_count = validation_count
    else:
        coarse_count = min(config.joint_search_coarse_samples, validation_count)
        final_count = min(config.joint_search_final_samples, validation_count)
    fit_inputs = student_inputs[:fit_count]
    validation_inputs = student_inputs[validation_start : validation_start + validation_count]
    teacher_attention_fit = teacher_targets.attention[:fit_count]
    teacher_attention_validation = teacher_targets.attention[
        validation_start : validation_start + validation_count
    ]
    teacher_mlp_fit = teacher_targets.mlp[:fit_count]
    teacher_mlp_validation = teacher_targets.mlp[
        validation_start : validation_start + validation_count
    ]
    teacher_block_validation = teacher_outputs[
        validation_start : validation_start + validation_count
    ].half().cpu()
    context = torch.cat(quantizers["self_attn.o_proj"].inputs).to(config.device)
    families = config.vo_rotation_candidates if config.rotation == "vo_foldable" else ("identity",)
    rotations = make_vo_rotation_candidates(
        layer.self_attn, context, families, config.rotation_candidate_seeds
    )
    ranks = config.functional_low_rank_ranks if config.low_rank_mode == "functional_branch" else (0,)
    salient_counts = (5, 7, 9, 11, 13) if config.joint_rotation_low_rank_search else (13,)
    baseline_o_bits = local_candidate_parameter_bits(o_module, config.blocksize, 13, 0, config.low_rank_dtype, config.low_rank_int8_scale_mode)
    baseline_down_bits = local_candidate_parameter_bits(down_module, config.blocksize, 13, 0, config.low_rank_dtype, config.low_rank_int8_scale_mode)
    baseline_budget = baseline_o_bits + baseline_down_bits
    attention_candidates: list[BranchCandidate] = []
    baseline_attention_name = "identity_o13_r0"
    rotation_time = 0.0
    low_rank_time = 0.0
    for rotation in rotations:
        rotation_started = time.perf_counter()
        folded_v, folded_o = fold_vo_weights(
            original_v, original_o, rotation,
            int(layer.self_attn.num_heads),
            int(getattr(layer.self_attn, "num_key_value_heads", layer.self_attn.num_heads)),
            int(layer.self_attn.head_dim),
        )
        v_result = _candidate_quantizer(quantizers["self_attn.v_proj"], v_module, folded_v, 13)
        v_result.artifact["vo_rotation"] = rotation.name
        v_module.weight.copy_(v_result.weight)
        o_module.weight.copy_(folded_o)
        remove_functional_adapter(o_module)
        o_source = _collect_quantizer(
            layer, o_module, "self_attn.o_proj", fit_inputs, kwargs, run_layer, config
        )
        rotation_time += time.perf_counter() - rotation_started
        for salient in salient_counts:
            o_result = _candidate_quantizer(o_source, o_module, folded_o, salient)
            o_module.weight.copy_(o_result.weight)
            o_result.artifact["vo_rotation"] = rotation.name
            remove_functional_adapter(o_module)
            z_fit, rank0_fit, _ = _capture_branch(
                layer, o_module, layer.self_attn, fit_inputs, kwargs, run_layer
            )
            lr_started = time.perf_counter()
            payloads = _fit_rank_payloads(
                z_fit, teacher_attention_fit, rank0_fit, ranks, config, "self_attn.o_proj"
            )
            low_rank_time += time.perf_counter() - lr_started
            for rank in ranks:
                bits = local_candidate_parameter_bits(
                    o_module, config.blocksize, salient, rank, config.low_rank_dtype,
                    config.low_rank_int8_scale_mode
                )
                if bits > baseline_budget or rank not in payloads:
                    continue
                payload = payloads[rank]
                install_functional_adapter(o_module, payload)
                _, branch_validation, _ = _capture_branch(
                    layer, o_module, layer.self_attn,
                    validation_inputs[:coarse_count], kwargs, run_layer,
                )
                loss = _branch_loss(
                    branch_validation,
                    teacher_attention_validation[:coarse_count], config,
                )
                name = f"{rotation.name}_o{salient}_r{rank}"
                attention_candidates.append(BranchCandidate(
                    name, loss, bits, salient, rank,
                    v_result.weight.detach().half().cpu(), deepcopy(v_result.artifact),
                    o_result.weight.detach().half().cpu(), deepcopy(o_result.artifact),
                    deepcopy(payload), rotation.name,
                ))
                remove_functional_adapter(o_module)
    attention_top = _forced_topk(
        attention_candidates, baseline_attention_name, config.functional_candidate_topk
    )
    baseline_attention = next(item for item in attention_candidates if item.name == baseline_attention_name)

    def apply_attention(candidate: BranchCandidate) -> None:
        v_module.weight.copy_(candidate.primary_weight.to(v_module.weight))
        o_module.weight.copy_(candidate.secondary_weight.to(o_module.weight))
        install_functional_adapter(o_module, candidate.low_rank_payload or {})

    apply_attention(baseline_attention)
    down_source = _collect_quantizer(
        layer, down_module, "mlp.down_proj", fit_inputs, kwargs, run_layer, config
    )
    mlp_candidates: list[BranchCandidate] = []
    baseline_mlp_name = "down13_r0"
    for salient in salient_counts:
        down_result = _candidate_quantizer(down_source, down_module, original_down, salient)
        down_module.weight.copy_(down_result.weight)
        remove_functional_adapter(down_module)
        z_fit, rank0_fit, _ = _capture_branch(
            layer, down_module, layer.mlp, fit_inputs, kwargs, run_layer
        )
        lr_started = time.perf_counter()
        payloads = _fit_rank_payloads(
            z_fit, teacher_mlp_fit, rank0_fit, ranks, config, "mlp.down_proj"
        )
        low_rank_time += time.perf_counter() - lr_started
        for rank in ranks:
            bits = local_candidate_parameter_bits(
                down_module, config.blocksize, salient, rank, config.low_rank_dtype,
                config.low_rank_int8_scale_mode
            )
            if bits > baseline_budget or rank not in payloads:
                continue
            payload = payloads[rank]
            install_functional_adapter(down_module, payload)
            _, branch_validation, _ = _capture_branch(
                layer, down_module, layer.mlp,
                validation_inputs[:coarse_count], kwargs, run_layer,
            )
            loss = _branch_loss(branch_validation, teacher_mlp_validation[:coarse_count], config)
            name = f"down{salient}_r{rank}"
            mlp_candidates.append(BranchCandidate(
                name, loss, bits, salient, rank,
                down_result.weight.detach().half().cpu(), deepcopy(down_result.artifact),
                low_rank_payload=deepcopy(payload),
            ))
            remove_functional_adapter(down_module)
    mlp_top = _forced_topk(mlp_candidates, baseline_mlp_name, config.functional_candidate_topk)

    def apply_mlp(candidate: BranchCandidate) -> None:
        down_module.weight.copy_(candidate.primary_weight.to(down_module.weight))
        install_functional_adapter(down_module, candidate.low_rank_payload or {})

    combinations: list[tuple[float, BranchCandidate, BranchCandidate]] = []
    for attention_candidate in attention_top:
        for mlp_candidate in mlp_top:
            if attention_candidate.parameter_bits + mlp_candidate.parameter_bits > baseline_budget:
                continue
            apply_attention(attention_candidate)
            apply_mlp(mlp_candidate)
            student = run_layer(layer, validation_inputs[:final_count], kwargs)
            target = teacher_block_validation[:final_count].to(student.device)
            weights = geometry_weights(target, config.geometry_loss, config.geometry_gamma, config.geometry_eps)
            loss = float(geometry_loss(student, target, weights))
            combinations.append((loss, attention_candidate, mlp_candidate))
    if not combinations:
        raise RuntimeError("no joint candidate satisfies the local baseline bit budget")
    final_loss, selected_attention, selected_mlp = min(combinations, key=lambda item: item[0])
    apply_attention(selected_attention)
    apply_mlp(selected_mlp)
    selected_attention.secondary_artifact["low_rank"] = selected_attention.low_rank_payload or {}
    selected_mlp.primary_artifact["low_rank"] = selected_mlp.low_rank_payload or {}
    install_activation_quantizer(v_module, selected_attention.primary_artifact["activation"])
    install_activation_quantizer(o_module, selected_attention.secondary_artifact["activation"])
    install_activation_quantizer(down_module, selected_mlp.primary_artifact["activation"])
    results = {
        "self_attn.v_proj": QuantizationResult(v_module.weight.detach(), selected_attention.primary_artifact, 0.0),
        "self_attn.o_proj": QuantizationResult(o_module.weight.detach(), selected_attention.secondary_artifact, 0.0),
        "mlp.down_proj": QuantizationResult(down_module.weight.detach(), selected_mlp.primary_artifact, 0.0),
    }
    statistics: dict[str, float | str] = {
        "vo_rotation": selected_attention.rotation,
        "o_rank": float(selected_attention.rank),
        "o_salient_columns": float(selected_attention.salient_columns),
        "down_rank": float(selected_mlp.rank),
        "down_salient_columns": float(selected_mlp.salient_columns),
        "joint_validation_loss": final_loss,
        "joint_parameter_bits": float(selected_attention.parameter_bits + selected_mlp.parameter_bits),
        "joint_baseline_bits": float(baseline_budget),
        "rotation_search_time_s": rotation_time,
        "functional_low_rank_time_s": low_rank_time,
        "joint_search_time_s": time.perf_counter() - started,
        "functional_lr_alternating_accepts": 0.0,
        "functional_lr_alternating_mean_branch_reduction": 0.0,
        "functional_lr_alternating_mean_block_reduction": 0.0,
        "topup_accepted_upgrades": 0.0,
        "topup_parameter_bits_before": float(selected_attention.parameter_bits + selected_mlp.parameter_bits),
        "topup_parameter_bits_after": float(selected_attention.parameter_bits + selected_mlp.parameter_bits),
        "selective_rotation_candidates": 0.0,
        "selective_rotation_accepts": 0.0,
    }
    del attention_candidates, mlp_candidates, context
    torch.cuda.empty_cache()
    return JointSearchResult(results, statistics)

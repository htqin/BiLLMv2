"""Teacher/student layer-by-layer quantization."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor, nn

from billmv2.baseline import find_layers
from billmv2.config import BiLLMv2Config
from billmv2.quantization.activation import install_activation_quantizer
from billmv2.quantization.joint_search import (
    BranchTargets,
    capture_teacher_branches,
    search_block_rotation_low_rank,
)
from billmv2.quantization.v2_quantizer import BiLLMv2Quantizer
from billmv2.reconstruction.geometry import SequentialStreams
from billmv2.reconstruction.objectives import geometry_loss, geometry_weights

LOGGER = logging.getLogger(__name__)


def _prepare_coupled_saliency(
    quantizers: dict[str, BiLLMv2Quantizer], config: BiLLMv2Config
) -> dict[str, int]:
    """Select independent or shared salient scores with a local loss."""

    enabled_groups = []
    if config.coupled_saliency in {"coupled_qkv", "coupled_qkv_gate_up"}:
        enabled_groups.append(("qkv", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")))
    if config.coupled_saliency in {"coupled_gate_up", "coupled_qkv_gate_up"}:
        enabled_groups.append(("gate_up", ("mlp.gate_proj", "mlp.up_proj")))
    statistics: dict[str, int] = {}
    for group_name, suffixes in enabled_groups:
        members = [
            quantizers[name] for suffix in suffixes
            for name in quantizers if name.endswith(suffix)
        ]
        if len(members) != len(suffixes):
            continue
        scores = {
            mixing: [member.initial_salient_scores(mixing) for member in members]
            for mixing in config.saliency_l2_lambdas
        }
        for start in scores[config.saliency_l2_lambdas[0]][0]:
            candidates: list[tuple[float, str, list[Tensor]]] = []
            for mixing, member_scores in scores.items():
                independent = [value[start] for value in member_scores]
                independent_loss = sum(
                    member.evaluate_salient_scores(start, score)
                    for member, score in zip(members, independent)
                )
                candidates.append((independent_loss, f"{group_name}_independent_l{mixing:g}", independent))
                coupled = torch.stack(independent).sum(dim=0)
                coupled_loss = sum(
                    member.evaluate_salient_scores(start, coupled) for member in members
                )
                candidates.append(
                    (coupled_loss, f"{group_name}_coupled_l{mixing:g}", [coupled] * len(members))
                )
            _, choice, selected = min(candidates, key=lambda item: item[0])
            statistics[choice] = statistics.get(choice, 0) + 1
            for member, score in zip(members, selected):
                member.salient_score_overrides[start] = score
    return statistics


class _CaptureInput(RuntimeError):
    pass


def _model_parts(model: nn.Module) -> tuple[Any, list[nn.Module], list[nn.Module]]:
    if model.config.model_type == "opt":
        decoder = model.model.decoder
        prefixes = [decoder.embed_tokens, decoder.embed_positions]
        for name in ("project_in", "project_out"):
            module = getattr(decoder, name, None)
            if module is not None:
                prefixes.append(module)
        return decoder, decoder.layers, prefixes
    if "llama" in model.config.model_type:
        return model.model, model.model.layers, [model.model.embed_tokens, model.model.norm]
    raise ValueError(f"unsupported model type: {model.config.model_type}")


def _run_layer(layer: nn.Module, inputs: Tensor, kwargs: dict[str, Any]) -> Tensor:
    outputs = torch.zeros_like(inputs)
    for index in range(inputs.shape[0]):
        current_kwargs = {
            name: value
            for name, value in kwargs.items()
            if value is not None
            and name in {"attention_mask", "position_ids", "position_embeddings"}
        }
        outputs[index] = layer(inputs[index].unsqueeze(0), **current_kwargs)[0]
    return outputs


@torch.no_grad()
def capture_first_layer_inputs(
    model: nn.Module,
    dataloader: Iterable[Any],
    config: BiLLMv2Config,
) -> tuple[Tensor, dict[str, Any]]:
    """Capture prefix activations using BiLLM's memory-bounded catcher pattern."""

    _, layers, prefixes = _model_parts(model)
    device = torch.device(config.device)
    for module in prefixes:
        module.to(device)
    layers[0].to(device)
    dtype = next(model.parameters()).dtype
    inputs = torch.zeros(
        (config.nsamples, model.seqlen, model.config.hidden_size),
        device=device,
        dtype=dtype,
    )
    cache: dict[str, Any] = {"index": 0}

    class Catcher(nn.Module):
        def __init__(self, module: nn.Module) -> None:
            super().__init__()
            self.module = module

        def forward(self, hidden_states: Tensor, **kwargs: Any) -> Tensor:
            inputs[cache["index"]] = hidden_states
            cache["index"] += 1
            cache.update(kwargs)
            raise _CaptureInput

    layers[0] = Catcher(layers[0])
    try:
        for batch in dataloader:
            if cache["index"] >= config.nsamples:
                break
            token_ids = batch[0] if isinstance(batch, (tuple, list)) else batch
            try:
                model(token_ids.to(device))
            except _CaptureInput:
                continue
    finally:
        layers[0] = layers[0].module
    if cache["index"] != config.nsamples:
        raise RuntimeError(f"captured {cache['index']} samples, expected {config.nsamples}")
    layers[0].cpu()
    for module in prefixes:
        module.cpu()
    torch.cuda.empty_cache()
    return inputs, cache


@torch.no_grad()
def quantize_sequential(
    model: nn.Module,
    dataloader: Iterable[Any],
    config: BiLLMv2Config,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Quantize blocks with distinct full-precision teacher and quantized student streams."""

    use_cache = model.config.use_cache
    model.config.use_cache = False
    initial, kwargs = capture_first_layer_inputs(model, dataloader, config)
    streams = SequentialStreams(initial, initial.clone())
    _, layers, _ = _model_parts(model)
    artifacts: dict[str, Any] = {}
    geometry_metrics: dict[str, float] = {}
    try:
        layer_count = len(layers) if config.max_layers == 0 else min(config.max_layers, len(layers))
        for layer_index, cpu_layer in list(enumerate(layers))[:layer_count]:
            layer = cpu_layer.to(config.device)
            joint_search = (
                config.joint_rotation_low_rank_search
                or config.rotation == "vo_foldable"
                or config.low_rank_mode == "functional_branch"
            )
            teacher_targets: BranchTargets | None = None
            if joint_search:
                target_samples = max(
                    config.rotation_fit_samples + config.rotation_validation_samples,
                    config.functional_fit_samples + config.functional_validation_samples,
                    config.joint_search_final_samples,
                )
                teacher_outputs, teacher_targets = capture_teacher_branches(
                    layer, streams.teacher_inputs, kwargs, _run_layer, target_samples
                )
            else:
                teacher_outputs = _run_layer(layer, streams.teacher_inputs, kwargs)
            subset = find_layers(layer)
            quantizers: dict[str, BiLLMv2Quantizer] = {}
            for name, module in subset.items():
                module_selected = (
                    not config.target_modules
                    or any(name.endswith(target) for target in config.target_modules)
                )
                selected = (
                    config.minlayer <= layer_index < config.maxlayer
                    and config.quant_only in name
                    and module_selected
                )
                if selected == config.invert:
                    continue
                quantizers[name] = BiLLMv2Quantizer(module, config, name)
            observer_handles = []
            for name, module in subset.items():
                if name in quantizers:
                    observer_handles.append(
                        module.register_forward_pre_hook(
                            lambda _module, args, quantizer=quantizers[name]:
                            quantizer.observe_activation(args[0])
                        )
                    )
            try:
                _run_layer(layer, streams.student_inputs, kwargs)
            finally:
                for handle in observer_handles:
                    handle.remove()
            for quantizer in quantizers.values():
                quantizer.finalize_activation()
            hessian_handles = []
            for name, module in subset.items():
                if name not in quantizers:
                    continue
                quantizer = quantizers[name]

                def quantize_and_collect(
                    _module: nn.Module,
                    args: tuple[Tensor, ...],
                    current: BiLLMv2Quantizer = quantizer,
                ) -> tuple[Tensor]:
                    quantized_inputs = current.quantize_activation(args[0])
                    current.add_batch(quantized_inputs)
                    return (quantized_inputs,)

                hessian_handles.append(module.register_forward_pre_hook(quantize_and_collect))
            try:
                _run_layer(layer, streams.student_inputs, kwargs)
            finally:
                for handle in hessian_handles:
                    handle.remove()
            saliency_statistics = _prepare_coupled_saliency(quantizers, config)
            for choice, count in saliency_statistics.items():
                geometry_metrics[f"saliency_{layer_index}_{choice}"] = float(count)
            layer_round2_accepted = 0
            layer_round2_rejected = 0
            layer_accepted_reduction = 0.0
            searched_names = {
                "self_attn.v_proj", "self_attn.o_proj", "mlp.down_proj"
            } if joint_search else set()
            selected_results: dict[str, Any] = {}
            for name, quantizer in quantizers.items():
                if name in searched_names:
                    continue
                result = quantizer.quantize()
                selected_results[name] = result
                install_activation_quantizer(subset[name], result.artifact["activation"])
                LOGGER.info("quantized fixed model.layers.%d.%s, OBC error %.6g", layer_index, name, result.error)
            if joint_search:
                if teacher_targets is None:
                    raise RuntimeError("joint search is missing teacher branch targets")
                search_result = search_block_rotation_low_rank(
                    layer, quantizers, streams.student_inputs, teacher_outputs,
                    teacher_targets, kwargs, _run_layer, config,
                )
                selected_results.update(search_result.results)
                for key, value in search_result.statistics.items():
                    geometry_metrics[f"joint_{layer_index}_{key}"] = value
            for name, result in selected_results.items():
                layer_round2_accepted += result.round2_accepted
                layer_round2_rejected += result.round2_rejected
                layer_accepted_reduction += result.accepted_loss_reduction
                full_name = f"model.layers.{layer_index}.{name}"
                if model.config.model_type == "opt":
                    full_name = f"model.decoder.layers.{layer_index}.{name}"
                artifacts[full_name] = result.artifact
                for block in result.artifact["blocks"]:
                    family = str(block.get("split_family", "asymmetric"))
                    key = f"split_family_{layer_index}_{family}"
                    geometry_metrics[key] = geometry_metrics.get(key, 0.0) + 1.0
                install_activation_quantizer(subset[name], result.artifact["activation"])
                LOGGER.info("quantized %s, OBC error %.6g", full_name, result.error)
            geometry_metrics[f"round2_accepted_{layer_index}"] = float(layer_round2_accepted)
            geometry_metrics[f"round2_rejected_{layer_index}"] = float(layer_round2_rejected)
            geometry_metrics[f"round2_reduction_sum_{layer_index}"] = layer_accepted_reduction
            student_outputs = _run_layer(layer, streams.student_inputs, kwargs)
            metric = geometry_weights(
                teacher_outputs,
                config.geometry_loss,
                config.geometry_gamma,
                config.geometry_eps,
            )
            geometry_metrics[str(layer_index)] = float(
                geometry_loss(student_outputs, teacher_outputs, metric)
            )
            streams.advance(teacher_outputs, student_outputs)
            layers[layer_index] = layer.cpu()
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                geometry_metrics[f"memory_mb_{layer_index}"] = (
                    torch.cuda.memory_allocated(config.device) / 1024**2
                )
    finally:
        model.config.use_cache = use_cache
    return artifacts, geometry_metrics

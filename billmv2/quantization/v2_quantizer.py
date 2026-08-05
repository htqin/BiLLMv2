"""BiLLM-v2 tensor quantizer with GPTQ/OBC error propagation."""

from __future__ import annotations

from dataclasses import dataclass
import logging

import torch
from torch import Tensor, nn

from billmv2.config import BiLLMv2Config
from billmv2.low_rank.compensation import LowRankFactors, weighted_low_rank
from billmv2.transforms.rotation import make_block_rotation
from billmv2.utils.bits import pack_bits, pack_indices

from .activation import ActivationQuantizer
from .binarizer import BinaryApproximation, binary_approximation
from .residual_selector import salient_scores, select_salient_mask
from .splitting import SplitResult, adaptive_split

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class QuantizationResult:
    """Return quantized weights, compact metadata, and weighted error."""

    weight: Tensor
    artifact: dict[str, object]
    error: float
    round2_accepted: int = 0
    round2_rejected: int = 0
    accepted_loss_reduction: float = 0.0


def _pack_active_signs(signs: Tensor, mask: Tensor) -> dict[str, object]:
    """Pack only signs selected by a branch mask."""

    return pack_bits(signs.permute(1, 0, 2)[:, mask] > 0)


def _factor_payload(factors: LowRankFactors, storage_dtype: str) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    if factors.rank == 0:
        return factors.u, factors.v, {}
    if storage_dtype == "fp16":
        stored_u = factors.u.half()
        stored_v = factors.v.half()
        return (
            stored_u.to(factors.u.dtype),
            stored_v.to(factors.v.dtype),
            {"u": stored_u.cpu(), "v": stored_v.cpu()},
        )
    if storage_dtype != "int8":
        raise ValueError(f"unsupported low-rank dtype: {storage_dtype}")
    payload: dict[str, Tensor] = {}
    dequantized: list[Tensor] = []
    for name, factor in (("u", factors.u), ("v", factors.v)):
        scale = factor.abs().max().clamp_min(1e-12) / 127
        quantized = torch.round(factor / scale).clamp(-127, 127).to(torch.int8)
        payload[name] = quantized.cpu()
        payload[f"{name}_scale"] = scale.float().cpu()
        dequantized.append(quantized.to(factor.dtype) * scale)
    return dequantized[0], dequantized[1], payload


class BiLLMv2Quantizer:
    """Quantize one linear layer while preserving BiLLM's structured branches."""

    def __init__(
        self, layer: nn.Linear, config: BiLLMv2Config, module_name: str = ""
    ) -> None:
        if not isinstance(layer, nn.Linear):
            raise TypeError("BiLLMv2Quantizer currently supports nn.Linear")
        self.layer = layer
        self.config = config
        self.module_name = module_name
        self.hessian = torch.zeros(
            (layer.in_features, layer.in_features), device=layer.weight.device, dtype=torch.float32
        )
        self.num_tokens = 0
        self.num_samples = 0
        self.inputs: list[Tensor] = []
        self.salient_score_overrides: dict[int, Tensor] = {}
        self.activation = ActivationQuantizer(
            config.activation_bits,
            config.activation_group_size,
            config.activation_symmetric,
            config.activation_clip_method,
            config.rotation,
            config.rotation_block_size,
            config.seed,
        )

    def observe_activation(self, inputs: Tensor) -> None:
        """Observe full-precision student inputs for static activation calibration."""

        self.activation.observe(inputs)

    def finalize_activation(self) -> None:
        """Finalize persistent activation scales."""

        self.activation.finalize()

    def quantize_activation(self, inputs: Tensor) -> Tensor:
        """Return the configured reference activation fake quantization."""

        return self.activation.fake_quant(inputs)

    def add_batch(self, inputs: Tensor) -> None:
        """Accumulate the unnormalized input Hessian online."""

        if self.activation.scales is None:
            self.activation.observe(inputs)
        if inputs.shape[-1] != self.layer.in_features:
            raise ValueError("input feature dimension does not match layer")
        flat = inputs.detach().float().reshape(-1, inputs.shape[-1])
        batch_samples = int(inputs.shape[0]) if inputs.ndim >= 3 else 1
        self.hessian *= self.num_samples / (self.num_samples + batch_samples)
        self.num_samples += batch_samples
        scaled = flat * (2.0 / self.num_samples) ** 0.5
        self.hessian += scaled.transpose(0, 1) @ scaled
        self.num_tokens += flat.shape[0]
        if sum(item.shape[0] for item in self.inputs) < 4096:
            self.inputs.append(flat[: min(flat.shape[0], 4096)].cpu())

    def _quantize_block(
        self,
        weight: Tensor,
        hessian: Tensor,
        inputs: Tensor | None,
        inverse_hessian: Tensor | None = None,
        reference_quantized: Tensor | None = None,
        salient_score_override: Tensor | None = None,
        salient_fraction: float | None = None,
    ) -> tuple[Tensor, dict[str, object]]:
        diagonal = torch.diag(hessian).clamp_min(1e-8)
        core = weight
        core_base = binary_approximation(core, hessian_diag=diagonal, order=1)
        scoring_residual = (
            core - core_base.quantized
            if reference_quantized is None
            else core - reference_quantized
        )
        scores = (
            salient_score_override
            if salient_score_override is not None
            else salient_scores(
                core,
                scoring_residual,
                diagonal,
                self.config.salient_metric,
            )
        )
        salient = select_salient_mask(
            scores,
            tuple(core.shape),
            self.config.salient_fraction if salient_fraction is None else salient_fraction,
        )
        salient_approx = binary_approximation(core, salient, diagonal, order=2)
        split = adaptive_split(
            core,
            ~salient,
            diagonal,
            self.config.split_mode,
            self.config.split_candidates,
            self.config.split_rerank_topk,
            inputs,
            self.config.split_granularity,
            self.config.row_split_rerank,
            self.config.split_row_tile,
        )
        stored_salient_scales = salient_approx.scales.half()
        quantized = (
            salient_approx.signs * stored_salient_scales.to(weight).unsqueeze(-1)
        ).sum(dim=1) * salient
        index_dtype = torch.uint8 if core.shape[1] <= 256 else torch.int32
        artifact: dict[str, object] = {
            "salient_indices": pack_indices(
                torch.nonzero(salient[0], as_tuple=False).flatten(), core.shape[1]
            ),
            "salient_signs": _pack_active_signs(salient_approx.signs, salient),
            "salient_scales": stored_salient_scales.cpu(),
            "sparse_mask": pack_bits(split.sparse_mask),
            "split_family": split.family,
        }
        if split.candidate_histogram is not None:
            artifact["split_diagnostics"] = {
                "candidate_histogram": split.candidate_histogram,
                "row_diversity_ratio": split.row_diversity_ratio,
                "mean_row_loss_reduction": split.mean_row_loss_reduction,
                "boundary_candidate_ratio": split.boundary_candidate_ratio,
            }
        # Store branch signs/scales, not dense reconstructed weights.
        for name, mask in (
            ("concentrated", split.concentrated_mask),
            ("sparse", split.sparse_mask),
        ):
            approximation: BinaryApproximation = binary_approximation(core, mask, diagonal, order=1)
            stored_scales = approximation.scales.half()
            quantized += (
                approximation.signs * stored_scales.to(weight).unsqueeze(-1)
            ).sum(dim=1) * mask
            artifact[f"{name}_signs"] = _pack_active_signs(approximation.signs, mask)
            artifact[f"{name}_scales"] = stored_scales.cpu()
        return quantized, artifact

    def _fixed_bpw_factors(
        self,
        working: Tensor,
        hessian: Tensor,
        inputs: Tensor | None,
        inverse_factor: Tensor,
    ) -> tuple[LowRankFactors, float, float]:
        """Select a residual rank/saliency pair under the exact parameter budget."""

        rows, columns = working.shape
        provisional = torch.empty_like(working)
        for start in range(0, columns, self.config.blocksize):
            end = min(start + self.config.blocksize, columns)
            provisional[:, start:end], _ = self._quantize_block(
                working[:, start:end], hessian[start:end, start:end],
                None if inputs is None else inputs[:, start:end],
                inverse_factor[start:end, start:end],
                salient_fraction=self.config.salient_fraction,
            )
        residual = working - provisional
        del provisional
        max_factors = weighted_low_rank(
            residual, min(4, min(residual.shape)), self.config.low_rank_metric, hessian
        )
        sample_starts = list(range(0, columns, self.config.blocksize))
        if len(sample_starts) > 8:
            positions = torch.linspace(0, len(sample_starts) - 1, 8).round().long().tolist()
            sample_starts = [sample_starts[index] for index in positions]
        best: tuple[float, int, float] | None = None
        for rank in (0, 2, 4):
            if rank > max_factors.rank:
                continue
            low_rank = (
                None
                if rank == 0
                else max_factors.u[:, :rank] @ max_factors.v[:, :rank].transpose(0, 1)
            )
            core = working if low_rank is None else working - low_rank
            for fraction in (0.04, 0.06, 0.08, 0.10):
                selected = sum(
                    round(fraction * min(self.config.blocksize, columns - start))
                    for start in range(0, columns, self.config.blocksize)
                )
                parameter_bits = rows * columns + rows * selected
                parameter_bits += (
                    16 * rank * (rows + columns)
                    if self.config.low_rank_dtype == "fp16"
                    else 8 * rank * (rows + columns) + 64
                )
                parameter_bpw = parameter_bits / (rows * columns)
                if parameter_bpw > self.config.fixed_bpw_target + 1e-12:
                    continue
                loss_sum = 0.0
                element_count = 0
                for start in sample_starts:
                    end = min(start + self.config.blocksize, columns)
                    candidate, _ = self._quantize_block(
                        core[:, start:end], hessian[start:end, start:end],
                        None if inputs is None else inputs[:, start:end],
                        inverse_factor[start:end, start:end],
                        salient_fraction=fraction,
                    )
                    difference = core[:, start:end] - candidate
                    loss_sum += float(
                        (difference.square() * torch.diag(
                            hessian[start:end, start:end]
                        ).unsqueeze(0)).sum()
                    )
                    element_count += difference.numel()
                score = loss_sum / element_count
                if best is None or score < best[0]:
                    best = (score, rank, fraction)
            del core, low_rank
        if best is None:
            raise RuntimeError("no fixed-BPW low-rank candidate satisfies the budget")
        _, selected_rank, selected_fraction = best
        factors = LowRankFactors(
            max_factors.u[:, :selected_rank], max_factors.v[:, :selected_rank]
        )
        selected_bits = rows * columns + rows * sum(
            round(selected_fraction * min(self.config.blocksize, columns - start))
            for start in range(0, columns, self.config.blocksize)
        ) + 16 * selected_rank * (rows + columns)
        selected_bpw = selected_bits / (rows * columns)
        LOGGER.info(
            "fixed-BPW allocation %s: rank=%d salient=%.2f parameter_bpw=%.9f",
            self.module_name, selected_rank, selected_fraction, selected_bpw,
        )
        return factors, selected_fraction, selected_bpw

    def initial_salient_scores(self, mixing: float) -> dict[int, Tensor]:
        """Return normalized residual/L2 column scores for coupling."""

        if mixing < 0.0 or mixing > 1.0:
            raise ValueError("mixing must be in [0, 1]")
        weight = self.layer.weight.detach().float()
        diagonal = torch.diag(self.hessian).clamp_min(1e-8)
        result: dict[int, Tensor] = {}
        for start in range(0, weight.shape[1], self.config.blocksize):
            end = min(start + self.config.blocksize, weight.shape[1])
            block = weight[:, start:end]
            block_diagonal = diagonal[start:end]
            base = binary_approximation(block, hessian_diag=block_diagonal, order=1)
            residual = salient_scores(
                block, block - base.quantized, block_diagonal, "residual_hessian"
            )
            column_l2 = block.square().sum(dim=0)
            residual = residual / residual.mean().clamp_min(1e-12)
            column_l2 = column_l2 / column_l2.mean().clamp_min(1e-12)
            result[start] = mixing * residual + (1.0 - mixing) * column_l2
        return result

    def evaluate_salient_scores(self, start: int, scores: Tensor) -> float:
        """Evaluate one salient-score candidate with the local Hessian objective."""

        end = min(start + self.config.blocksize, self.layer.in_features)
        weight = self.layer.weight.detach().float()[:, start:end]
        hessian = self.hessian[start:end, start:end]
        quantized, _ = self._quantize_block(
            weight, hessian, None, salient_score_override=scores
        )
        return float(
            ((weight - quantized).square() * torch.diag(hessian).unsqueeze(0)).mean()
        )

    @torch.no_grad()
    def quantize(self) -> QuantizationResult:
        """Run alternating structured quantization and blockwise OBC."""

        if self.num_tokens == 0:
            raise RuntimeError("add_batch must be called before quantize")
        if self.activation.scales is None:
            self.activation.finalize()
        original = self.layer.weight.detach().float()
        linear_rotation = (
            self.config.linear_basis_rotation
            if self.config.rotation == "linear_basis"
            else self.config.rotation
            if self.config.rotation in {"hadamard", "random_orthogonal"}
            else "none"
        )
        rotation = make_block_rotation(
            original.shape[1],
            linear_rotation,
            self.config.rotation_block_size,
            self.config.seed,
            original.device,
        )
        working = original @ rotation
        hessian = rotation.transpose(0, 1) @ self.hessian @ rotation
        diagonal_indices = torch.arange(hessian.shape[0], device=hessian.device)
        dead = torch.diag(hessian) == 0
        hessian[dead, dead] = 1
        working[:, dead] = 0
        damping = self.config.percdamp * torch.diag(hessian).mean()
        hessian[diagonal_indices, diagonal_indices] += damping.clamp_min(1e-8)
        inverse_factor = torch.linalg.cholesky(
            torch.cholesky_inverse(torch.linalg.cholesky(hessian)), upper=True
        )
        rotated_inputs = None
        if self.inputs:
            rotated_inputs = torch.cat(self.inputs).to(working) @ rotation
        fixed_target = self.config.fixed_bpw_low_rank and any(
            self.module_name.endswith(name)
            for name in ("q_proj", "v_proj", "o_proj", "down_proj")
        )
        selected_salient_fraction = self.config.salient_fraction
        selected_parameter_bpw = 1.0 + self.config.salient_fraction
        if fixed_target:
            factors, selected_salient_fraction, selected_parameter_bpw = (
                self._fixed_bpw_factors(working, hessian, rotated_inputs, inverse_factor)
            )
        elif self.config.low_rank_mode == "weight_residual" and self.config.low_rank_rank > 0:
            provisional = binary_approximation(
                working, hessian_diag=torch.diag(hessian), order=1
            )
            residual = working - provisional.quantized
            factors = weighted_low_rank(
                residual,
                min(self.config.low_rank_rank, min(residual.shape)),
                self.config.low_rank_metric,
                hessian,
            )
        else:
            factors = weighted_low_rank(working, 0, "weight")
        low_rank_u, low_rank_v, low_rank_payload = _factor_payload(
            factors, self.config.low_rank_dtype
        )
        global_low_rank = low_rank_u @ low_rank_v.transpose(0, 1)
        working -= global_low_rank
        quantized = torch.zeros_like(working)
        blocks: list[dict[str, object]] = []
        total_error = 0.0
        round2_accepted = 0
        round2_rejected = 0
        accepted_loss_reduction = 0.0
        for start in range(0, working.shape[1], self.config.blocksize):
            end = min(start + self.config.blocksize, working.shape[1])
            block_weight = working[:, start:end].clone()
            block_hessian = hessian[start:end, start:end]
            block_inputs = rotated_inputs[:, start:end] if rotated_inputs is not None else None
            block_quantized = block_weight
            block_artifact: dict[str, object] = {}
            best_loss = float("inf")
            round1_loss = float("inf")
            reference_quantized = None
            for alternating_index in range(self.config.alternating_steps):
                candidate_quantized, candidate_artifact = self._quantize_block(
                    block_weight,
                    block_hessian,
                    block_inputs,
                    inverse_factor[start:end, start:end],
                    reference_quantized,
                    (
                        self.salient_score_overrides.get(start)
                        if alternating_index == 0
                        else None
                    ),
                    selected_salient_fraction,
                )
                candidate_loss = float(
                    (
                        (block_weight - candidate_quantized).square()
                        * torch.diag(block_hessian).unsqueeze(0)
                    ).mean()
                )
                LOGGER.info(
                    "alternating block %d:%d round %d reconstruction loss %.9g",
                    start,
                    end,
                    alternating_index + 1,
                    candidate_loss,
                )
                reference_quantized = candidate_quantized
                if alternating_index == 0:
                    round1_loss = candidate_loss
                elif alternating_index == 1:
                    if candidate_loss < round1_loss:
                        round2_accepted += 1
                        accepted_loss_reduction += round1_loss - candidate_loss
                    else:
                        round2_rejected += 1
                if candidate_loss < best_loss:
                    best_loss = candidate_loss
                    block_quantized = candidate_quantized
                    block_artifact = candidate_artifact
            quantized[:, start:end] = block_quantized
            denominator = torch.diag(inverse_factor)[start:end].clamp_min(1e-8)
            error = (block_weight - block_quantized) / denominator.unsqueeze(0)
            total_error += float((error.square() / 2).sum())
            if end < working.shape[1] and not self.config.disable_gptq:
                working[:, end:] -= error @ inverse_factor[start:end, end:]
            block_artifact.update({"start": start, "end": end})
            blocks.append(block_artifact)
        restored = (quantized + global_low_rank) @ rotation.transpose(0, 1)
        self.layer.weight.copy_(restored.to(self.layer.weight))
        artifact = {
            "shape": list(original.shape),
            "rotation": linear_rotation,
            "rotation_matrix": None,
            "rotation_block_size": self.config.rotation_block_size,
            "rotation_seed": self.config.seed,
            "blocks": blocks,
            "low_rank": low_rank_payload,
            "selected_salient_fraction": selected_salient_fraction,
            "parameter_bpw": selected_parameter_bpw,
            "activation": self.activation.to_artifact(),
        }
        return QuantizationResult(
            restored.to(self.layer.weight),
            artifact,
            total_error,
            round2_accepted,
            round2_rejected,
            accepted_loss_reduction,
        )

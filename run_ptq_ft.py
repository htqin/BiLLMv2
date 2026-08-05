#!/usr/bin/env python3
"""Run PTQ followed by compact calibration-only refinement."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as functional

from billmv2.finetune.refinement import (
    RefinedLinear,
    RefinementConfig,
    export_refinement,
    freeze_and_wrap,
    trainable_parameter_ratio,
)
from billmv2.reconstruction.objectives import geometry_loss, geometry_weights
from billmv2.utils.bits import account_saved_artifact_storage, artifact_bpw_breakdown
from run_ptq import build_parser, run

LOGGER = logging.getLogger(__name__)


def build_ft_parser():
    """Extend the PTQ parser with compact refinement options."""

    parser = build_parser()
    for action in parser._actions:
        if action.dest == "rotation":
            action.choices = [
                "none",
                "hadamard",
                "random_orthogonal",
                "block_cayley",
                "householder",
            ]
    parser.add_argument("--ft_steps", type=int, default=200)
    parser.add_argument("--ft_lr", type=float, default=1e-4)
    parser.add_argument("--ft_weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--ft_optimize",
        nargs="+",
        choices=["rotation", "scales", "activation_scales", "low_rank", "split"],
        default=["rotation", "scales", "activation_scales", "low_rank"],
    )
    parser.add_argument("--ft_kl_weight", type=float, default=0.1)
    parser.add_argument("--ft_gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--ft_amp", action="store_true")
    parser.add_argument("--ft_sign_refresh_interval", type=int, default=0)
    return parser


def _merge_wrappers(model: torch.nn.Module) -> None:
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, RefinedLinear)
    ]
    for name, module in targets:
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, module.merged_linear())


def main() -> None:
    """Execute PTQ+ without retaining a full teacher model."""

    args = build_ft_parser().parse_args()
    learned_rotation = args.rotation
    if learned_rotation in {"block_cayley", "householder"}:
        args.rotation = "none"
    args.capture_fp_targets = True
    model, artifacts = run(args)
    refinement = RefinementConfig(
        steps=args.ft_steps,
        learning_rate=args.ft_lr,
        weight_decay=args.ft_weight_decay,
        optimize=tuple(args.ft_optimize),
        kl_weight=args.ft_kl_weight,
        gradient_accumulation_steps=args.ft_gradient_accumulation_steps,
        amp=args.ft_amp,
        rotation=learned_rotation if learned_rotation in {"block_cayley", "householder"} else "block_cayley",
        rank=max(args.low_rank_rank, 2),
        block_size=args.rotation_block_size,
    )
    fp_logits = model._billmv2_fp_logits
    fp_hidden = model._billmv2_fp_hidden
    calibration_loader = model._billmv2_calibration_loader
    del model._billmv2_fp_logits, model._billmv2_fp_hidden, model._billmv2_calibration_loader
    freeze_and_wrap(model, refinement, set(artifacts))
    trainable, total, ratio = trainable_parameter_ratio(model)
    LOGGER.info(
        "trainable parameters / total parameters: %d / %d (%.6f%%)",
        trainable,
        total,
        ratio * 100,
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=refinement.learning_rate, weight_decay=refinement.weight_decay
    )
    model.to(args.device).train()
    ft_metric = geometry_weights(
        torch.stack(fp_hidden).to(args.device),
        args.geometry_loss,
        args.geometry_gamma,
        args.geometry_eps,
    )
    optimizer.zero_grad(set_to_none=True)
    scaler = torch.cuda.amp.GradScaler(enabled=refinement.amp)
    for step in range(refinement.steps):
        index = step % len(calibration_loader)
        sample = calibration_loader[index]
        token_ids = sample[0] if isinstance(sample, (tuple, list)) else sample
        with torch.cuda.amp.autocast(enabled=refinement.amp):
            outputs = model(
                token_ids.to(args.device), output_hidden_states=True, use_cache=False
            )
            teacher_logits = fp_logits[index].to(args.device).float()
            student_logits = outputs.logits[:, -1].float()
            kl = functional.kl_div(
                functional.log_softmax(student_logits, dim=-1),
                functional.softmax(teacher_logits, dim=-1),
                reduction="batchmean",
            )
            teacher_hidden = fp_hidden[index].to(args.device)
            student_hidden = outputs.hidden_states[-1][:, -1]
            geo = geometry_loss(student_hidden, teacher_hidden, ft_metric)
            loss = (geo + refinement.kl_weight * kl) / refinement.gradient_accumulation_steps
        scaler.scale(loss).backward()
        if (step + 1) % refinement.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        if step == 0 or (step + 1) % 20 == 0:
            LOGGER.info("ft step %d/%d loss=%.6g", step + 1, refinement.steps, float(loss))
    output_dir = Path(args.output_dir)
    refinement_payload = export_refinement(model)
    torch.save(refinement_payload, output_dir / "refinement.pt")
    config_path = output_dir / "config.json"
    saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    saved_config["artifact_type"] = "ptq_ft"
    config_path.write_text(json.dumps(saved_config, indent=2), encoding="utf-8")
    bpw = account_saved_artifact_storage(
        artifact_bpw_breakdown(artifacts, refinement_payload),
        output_dir,
        include_refinement=True,
    )
    (output_dir / "bpw.json").write_text(json.dumps(bpw, indent=2), encoding="utf-8")
    if args.save_merged_model:
        _merge_wrappers(model)
        model.save_pretrained(output_dir / "merged_model_ft")


if __name__ == "__main__":
    main()

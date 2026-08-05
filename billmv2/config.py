"""Central configuration for BiLLM-v2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class BiLLMv2Config:
    """Configure a deterministic BiLLM-v2 run."""

    model: str
    calib_dataset: str = "c4"
    nsamples: int = 128
    seqlen: int = 0
    calib_candidate_size: int = 1024
    calib_selector: str = "hybrid"
    calib_feature: str = "joint"
    calib_feature_dim: int = 64
    calib_probe_stride: int = 4
    seed: int = 0
    device: str = "cuda:0"
    blocksize: int = 128
    percdamp: float = 0.01
    rotation: str = "none"
    rotation_block_size: int = 128
    linear_basis_rotation: str = "hadamard"
    vo_rotation_candidates: tuple[str, ...] = (
        "identity", "signed_hadamard", "random_orthogonal", "covariance_hadamard"
    )
    rotation_candidate_seeds: tuple[int, ...] = (0, 1)
    rotation_fit_samples: int = 32
    rotation_validation_samples: int = 16
    low_rank_rank: int = 0
    low_rank_mode: str = "none"
    low_rank_metric: str = "diag_hessian"
    low_rank_dtype: str = "fp16"
    low_rank_int8_scale_mode: str = "tensor"
    functional_low_rank_ranks: tuple[int, ...] = (0, 2, 4)
    functional_low_rank_max_rank: int = 4
    functional_lr_alternating_steps: int = 1
    functional_fit_samples: int = 0
    functional_validation_samples: int = 0
    functional_candidate_topk: int = 2
    functional_lookahead_margin: float = 0.005
    functional_lookahead_blocks: int = 1
    global_functional_budget_topup: bool = False
    target_parameter_bpw: float = 1.1015625
    topup_objective: str = "block_validation_gain_per_bit"
    selective_vo_rotation: bool = False
    max_rotation_rescue_blocks: int = 8
    rotation_acceptance_margin: float = 0.01
    low_rank_ridge: float = 1e-4
    low_rank_sketch_dim: int = 128
    low_rank_max_tokens: int = 4096
    joint_rotation_low_rank_search: bool = False
    joint_search_coarse_samples: int = 8
    joint_search_final_samples: int = 16
    joint_search_topk: int = 2
    fixed_bpw_low_rank: bool = False
    fixed_bpw_target: float = 1.101563
    activation_bits: int = 4
    activation_group_size: int = 128
    activation_symmetric: bool = True
    activation_clip_method: str = "mse"
    salient_metric: str = "residual_hessian"
    salient_fraction: float = 0.1
    coupled_saliency: str = "independent_residual_hessian"
    saliency_l2_lambdas: tuple[float, ...] = (1.0,)
    split_mode: str = "asymmetric"
    split_granularity: str = "global"
    split_candidates: int = 16
    split_rerank_topk: int = 4
    row_split_rerank: str = "none"
    split_row_tile: int = 256
    alternating_steps: int = 1
    geometry_loss: str = "diagonal_whiten"
    geometry_gamma: float = 0.5
    geometry_eps: float = 1e-5
    minlayer: int = -1
    maxlayer: int = 1000
    max_layers: int = 0
    target_modules: tuple[str, ...] = ()
    quant_only: str = ""
    invert: bool = False
    disable_gptq: bool = False
    output_dir: str = "outputs/default"
    save_merged_model: bool = False

    def __post_init__(self) -> None:
        if self.nsamples <= 0 or self.blocksize <= 0:
            raise ValueError("nsamples and blocksize must be positive")
        if self.seqlen < 0 or self.max_layers < 0:
            raise ValueError("seqlen and max_layers must be non-negative")
        if self.low_rank_rank < 0 or self.alternating_steps <= 0:
            raise ValueError("rank must be non-negative and alternating_steps positive")
        if self.fixed_bpw_target <= 0.0:
            raise ValueError("fixed_bpw_target must be positive")
        if self.low_rank_mode not in {"none", "weight_residual", "functional_branch"}:
            raise ValueError("unsupported low_rank_mode")
        allowed_functional_ranks = {0, 2, 4, 6, 8, 12}
        if any(rank not in allowed_functional_ranks for rank in self.functional_low_rank_ranks):
            raise ValueError("functional ranks must be one of 0, 2, 4, 6, 8, or 12")
        if self.functional_low_rank_max_rank not in allowed_functional_ranks:
            raise ValueError("functional_low_rank_max_rank must be one of 0, 2, 4, 6, 8, or 12")
        if max(self.functional_low_rank_ranks, default=0) > self.functional_low_rank_max_rank:
            raise ValueError("functional_low_rank_max_rank must cover all candidate ranks")
        if self.low_rank_int8_scale_mode not in {"tensor", "per_rank"}:
            raise ValueError("low_rank_int8_scale_mode must be tensor or per_rank")
        if self.functional_lr_alternating_steps not in {1, 2}:
            raise ValueError("functional_lr_alternating_steps must be 1 or 2")
        if min(self.functional_fit_samples, self.functional_validation_samples) < 0:
            raise ValueError("functional fit/validation sample counts must be non-negative")
        if self.functional_candidate_topk <= 0 or self.functional_lookahead_blocks < 0:
            raise ValueError("functional candidate topk/lookahead blocks must be valid")
        if self.functional_lookahead_margin < 0.0 or self.rotation_acceptance_margin < 0.0:
            raise ValueError("functional margins must be non-negative")
        if self.target_parameter_bpw <= 0.0 or self.max_rotation_rescue_blocks < 0:
            raise ValueError("target BPW and rescue block limit must be valid")
        if self.topup_objective != "block_validation_gain_per_bit":
            raise ValueError("unsupported top-up objective")
        if self.split_granularity not in {"global", "per_row"}:
            raise ValueError("split_granularity must be global or per_row")
        if self.row_split_rerank not in {"none", "linear_top2"}:
            raise ValueError("row_split_rerank must be none or linear_top2")
        if min(
            self.rotation_fit_samples, self.rotation_validation_samples,
            self.low_rank_sketch_dim, self.low_rank_max_tokens,
            self.joint_search_coarse_samples, self.joint_search_final_samples,
            self.joint_search_topk, self.split_row_tile,
        ) <= 0:
            raise ValueError("search sizes must be positive")
        if self.activation_bits not in {4, 8, 16}:
            raise ValueError("activation_bits must be 4, 8, or 16")
        if self.activation_group_size <= 0:
            raise ValueError("activation_group_size must be positive")
        if not 0.0 <= self.salient_fraction <= 1.0:
            raise ValueError("salient_fraction must be in [0, 1]")
        if not self.saliency_l2_lambdas or any(
            value < 0.0 or value > 1.0 for value in self.saliency_l2_lambdas
        ):
            raise ValueError("saliency_l2_lambdas must be non-empty and within [0, 1]")
        if self.geometry_eps <= 0.0:
            raise ValueError("geometry_eps must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable configuration values."""

        return asdict(self)

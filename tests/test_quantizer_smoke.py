import torch
from torch import nn

from billmv2.config import BiLLMv2Config
from billmv2.quantization.v2_quantizer import BiLLMv2Quantizer
from billmv2.utils.artifacts import reconstruct_weight


def test_tiny_linear_quantization_and_reconstruction() -> None:
    torch.manual_seed(5)
    layer = nn.Linear(16, 12, bias=False)
    config = BiLLMv2Config(
        model="tiny",
        nsamples=2,
        blocksize=8,
        rotation="hadamard",
        rotation_block_size=8,
        low_rank_rank=2,
        split_candidates=4,
        split_rerank_topk=2,
    )
    quantizer = BiLLMv2Quantizer(layer, config)
    quantizer.add_batch(torch.randn(2, 6, 16))
    result = quantizer.quantize()
    restored = reconstruct_weight(result.artifact)
    torch.testing.assert_close(restored, layer.weight.float(), atol=2e-3, rtol=2e-3)

import torch

from billmv2.transforms.rotation import make_block_rotation


def test_rotation_invariance() -> None:
    torch.manual_seed(1)
    inputs = torch.randn(7, 16)
    weight = torch.randn(9, 16)
    rotation = make_block_rotation(16, "hadamard", block_size=8)
    reference = inputs @ weight.transpose(0, 1)
    rotated = (inputs @ rotation) @ (weight @ rotation).transpose(0, 1)
    torch.testing.assert_close(reference, rotated, atol=1e-5, rtol=1e-5)

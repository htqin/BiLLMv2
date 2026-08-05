import torch

from billmv2.quantization.activation import (
    ActivationQuantizer,
    install_activation_quantizer,
)


def _calibrated(bits: int, width: int = 260) -> ActivationQuantizer:
    torch.manual_seed(5)
    quantizer = ActivationQuantizer(bits, group_size=128, symmetric=True, clip_method="mse")
    quantizer.observe(torch.randn(3, 7, width))
    quantizer.finalize()
    return quantizer


def test_symmetric_int4_range_and_group_shape() -> None:
    quantizer = _calibrated(4)
    inputs = torch.randn(2, 5, 260)
    output = quantizer.fake_quant(inputs)
    assert quantizer.scales is not None
    assert quantizer.scales.dtype == torch.float16
    assert quantizer.scales.shape == (3,)
    grouped = torch.nn.functional.pad(inputs, (0, 124)).reshape(2, 5, 3, 128)
    scales = quantizer.scales.float().view(1, 1, 3, 1)
    integers = torch.round(grouped / scales).clamp(-8, 7)
    assert integers.min() >= -8 and integers.max() <= 7
    expected = (integers * scales).reshape(2, 5, 384)[..., :260].to(inputs)
    torch.testing.assert_close(output, expected)


def test_activation_bits_16_is_identity() -> None:
    quantizer = _calibrated(16, width=32)
    inputs = torch.randn(2, 4, 32, dtype=torch.float16)
    output = quantizer.fake_quant(inputs)
    assert output.data_ptr() == inputs.data_ptr()
    torch.testing.assert_close(output, inputs, atol=0, rtol=0)


def test_artifact_roundtrip_is_exact() -> None:
    quantizer = _calibrated(4, width=128)
    restored = ActivationQuantizer.from_artifact(quantizer.to_artifact())
    inputs = torch.randn(2, 3, 128, dtype=torch.float16)
    torch.testing.assert_close(
        restored.fake_quant(inputs), quantizer.fake_quant(inputs), atol=0, rtol=0
    )


def test_linear_student_input_uses_int4_fake_quant() -> None:
    quantizer = _calibrated(4, width=128)
    linear = torch.nn.Linear(128, 8, bias=False)
    install_activation_quantizer(linear, quantizer.to_artifact())
    captured = []
    handle = linear.register_forward_pre_hook(
        lambda _module, arguments: captured.append(arguments[0].detach().clone())
    )
    inputs = torch.randn(2, 3, 128)
    linear(inputs)
    handle.remove()
    torch.testing.assert_close(captured[0], quantizer.fake_quant(inputs))
    assert not torch.equal(captured[0], inputs)

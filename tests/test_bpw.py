from billmv2.utils.bits import BitCount


def test_bit_count_matches_manual_total() -> None:
    count = BitCount(
        num_weights=100,
        binary_signs=100,
        masks=200,
        scales=64,
        means=32,
        thresholds=16,
        rotations=8,
        low_rank=80,
        metadata=12,
    )
    assert count.total_bits == 512
    assert count.strict_binary_bpw == 1.0
    assert count.effective_bpw == 5.12

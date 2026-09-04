"""Quantisation scales, and the dynamic-fixed-point variant.

The supervisor's point: at 11,786 parameters this model is small enough that
INT8 is the wrong default. Small networks have little redundancy to absorb
quantisation noise, and neither memory nor DSPs constrain this design -- so the
bits are cheap and a power-of-two scale, which turns requantisation into a
shift, is the thing worth having.

These tests pin the two properties that decision rests on.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Dynamic fixed point: power-of-two scales
# ---------------------------------------------------------------------------


def test_power_of_two_scale_rounds_up_never_down():
    """Down would make the scale too small for the observed maximum and clip
    it. The whole point of the format is that nothing saturates unexpectedly."""
    import torch

    from wearseizure.quant.scales import compute_symmetric_scale

    x = torch.tensor([0.0, 3.7])
    plain = compute_symmetric_scale(x, n_bits=8)
    pot = compute_symmetric_scale(x, n_bits=8, power_of_two=True)

    assert pot.scale >= plain.scale, "rounding down would clip the maximum"
    assert pot.is_power_of_two and not plain.is_power_of_two
    assert pot.scale < 2 * plain.scale, "at most one octave is given away"
    # Nothing saturates: the largest value still lands inside the code range.
    assert abs(pot.quantize(x)).max() <= pot.qmax


def test_power_of_two_costs_accuracy_at_equal_width():
    """Stated so it is not discovered later: dynamic fixed point at the SAME
    bit width is slightly worse than an arbitrary scale. It is chosen for the
    multiplier it removes from the requantisation path, and the accuracy is
    bought back by widening -- which is affordable when the model is 11,786
    parameters and memory is not the constraint."""
    import torch

    from wearseizure.quant.scales import compute_symmetric_scale

    torch.manual_seed(0)
    x = torch.randn(4096) * 3.7

    def err(bits, pot):
        q = compute_symmetric_scale(x, n_bits=bits, power_of_two=pot)
        return float((q.dequantize(q.quantize(x)) - x).abs().mean())

    assert err(8, True) > err(8, False), "power-of-two is worse at equal width"
    assert err(16, True) < err(8, False), "and widening more than buys it back"


def test_the_exponent_is_the_shift_a_datapath_would_use():
    import torch

    from wearseizure.quant.scales import compute_symmetric_scale

    q = compute_symmetric_scale(torch.tensor([1.0]), n_bits=8, power_of_two=True)
    assert q.scale == 2.0 ** q.exponent

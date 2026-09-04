"""Quantisation scales, and the dynamic-fixed-point variant.

The supervisor's point: at 11,786 parameters this model is small enough that
INT8 is the wrong default. Small networks have little redundancy to absorb
quantisation noise, and neither memory nor DSPs constrain this design -- so the
bits are cheap and a power-of-two scale, which turns requantisation into a
shift, is the thing worth having.

These tests pin the two properties that decision rests on.
"""
from __future__ import annotations

import math

import pytest

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


# ---------------------------------------------------------------------------
# Per-channel weight scales -- ladder step P1
# ---------------------------------------------------------------------------


def _skewed_depthwise_weight(spread: float = 128.0):
    """A depthwise weight whose channels span `spread`x in magnitude.

    Not an artificial worry: a depthwise layer has one independent filter per
    channel and nothing mixing them, so trained channels drift apart. This is
    the shape of the problem Krishnamoorthi (2018) measured on MobileNet.
    """
    import torch

    torch.manual_seed(0)
    n = 8
    gains = torch.logspace(0, math.log10(spread), n).view(n, 1, 1)
    return torch.randn(n, 1, 5) * gains


def _err(w, **kw):
    from wearseizure.quant.scales import compute_symmetric_scale

    q = compute_symmetric_scale(w, n_bits=8, **kw)
    return float((q.dequantize(q.quantize(w)) - w).abs().mean())


def test_per_channel_is_the_larger_lever_by_far():
    """The point of measuring both: the scale FORMAT (power-of-two or not) is a
    small effect next to the scale GRANULARITY. Choosing dynamic fixed point
    without also going per-channel would optimise the wrong axis."""
    w = _skewed_depthwise_weight()
    per_tensor = _err(w)
    per_channel = _err(w, per_channel=True)
    pow2_cost = _err(w, power_of_two=True) - per_tensor

    assert per_channel < per_tensor / 3, "per-channel must be a large win on skewed channels"
    assert pow2_cost < per_tensor - per_channel, "the format change is the smaller effect"


def test_per_channel_gives_one_scale_per_output_channel_shaped_to_broadcast():
    from wearseizure.quant.scales import compute_symmetric_scale

    w = _skewed_depthwise_weight()
    q = compute_symmetric_scale(w, n_bits=8, per_channel=True)
    assert q.per_channel
    assert q.scale.shape == (w.shape[0], 1, 1), "must broadcast against the weight"
    # Every channel uses its full code range, which is the whole point.
    used = q.quantize(w).abs().flatten(1).amax(dim=1)
    assert (used > q.qmax * 0.5).all()


def test_per_channel_power_of_two_gives_one_shift_per_channel():
    """This is what makes per-channel affordable in hardware: the requantisation
    stays a shift, it is just a different shift per output channel."""
    from wearseizure.quant.scales import compute_symmetric_scale

    q = compute_symmetric_scale(_skewed_depthwise_weight(), n_bits=8,
                                per_channel=True, power_of_two=True)
    assert q.is_power_of_two
    exps = q.exponent
    assert exps.shape == q.scale.shape
    assert exps.dtype.is_signed and not exps.dtype.is_floating_point


def test_a_scalar_scale_still_behaves_exactly_as_before():
    """Per-tensor is the default and must be untouched: every number in the
    experiment log was produced with it."""
    from wearseizure.quant.scales import compute_symmetric_scale

    w = _skewed_depthwise_weight()
    q = compute_symmetric_scale(w, n_bits=8)
    assert not q.per_channel
    assert isinstance(q.scale, float)
    assert q.scale == pytest.approx(float(w.abs().max()) / q.qmax)

"""Symmetric per-tensor quantization scale, shared by QAT fake-quant, PTQ
calibration and the integer reference so all three agree on what "scale" means.

Two families, and the difference is a hardware one:

  arbitrary scale   value = scale * q, with `scale` any float. Requantising
                    between layers needs a MULTIPLY and a shift.
  power-of-two      scale = 2**e, i.e. dynamic fixed point with a per-tensor
                    exponent. Requantising is a SHIFT alone, so the multiplier
                    disappears from that path entirely.

Power-of-two costs up to 2x of the representable range, because the scale is
rounded up to the next power of two, so at equal bit width it is slightly less
accurate. It is chosen for what it removes from the datapath, not for accuracy --
and on a model this small the bits it costs are cheap to buy back by widening.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class QuantScale:
    scale: float
    n_bits: int

    @property
    def is_power_of_two(self) -> bool:
        import math
        return self.scale > 0 and math.isclose(math.log2(self.scale) % 1.0, 0.0, abs_tol=1e-9)

    @property
    def exponent(self) -> int:
        """The shift amount a fixed-point datapath would use. Only meaningful
        for a power-of-two scale."""
        import math
        return round(math.log2(self.scale))

    @property
    def qmax(self) -> int:
        return 2 ** (self.n_bits - 1) - 1

    @property
    def qmin(self) -> int:
        return -(2 ** (self.n_bits - 1))

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(torch.round(x / self.scale), self.qmin, self.qmax)

    def dequantize(self, q: torch.Tensor) -> torch.Tensor:
        return q * self.scale


def _to_power_of_two(scale: float) -> float:
    """Round UP to the next power of two.

    Up, never to nearest: rounding down would make the scale too small for the
    observed maximum and clip it. Costs up to 2x of range, which is the price of
    turning the requantisation multiply into a shift.
    """
    import math
    return 2.0 ** math.ceil(math.log2(scale)) if scale > 0 else 1.0


def compute_symmetric_scale(
    x: torch.Tensor, n_bits: int = 8, power_of_two: bool = False
) -> QuantScale:
    qmax = 2 ** (n_bits - 1) - 1
    max_abs = float(x.detach().abs().max())
    scale = max_abs / qmax if max_abs > 0 else 1.0
    if power_of_two:
        scale = _to_power_of_two(scale)
    return QuantScale(scale=scale, n_bits=n_bits)


def compute_symmetric_scale_from_max(
    max_abs: torch.Tensor | float, n_bits: int = 8, power_of_two: bool = False
) -> QuantScale:
    qmax = 2 ** (n_bits - 1) - 1
    max_abs_f = float(max_abs)
    scale = max_abs_f / qmax if max_abs_f > 0 else 1.0
    if power_of_two:
        scale = _to_power_of_two(scale)
    return QuantScale(scale=scale, n_bits=n_bits)

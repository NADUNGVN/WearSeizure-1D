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
    """A scale, or one scale per output channel.

    `scale` is a float for per-tensor quantisation, or a tensor broadcastable
    against the weight for per-channel. Both go through the same quantize and
    dequantize, because division and multiplication broadcast -- a scalar is
    just the degenerate case, and keeping one class means the integer reference,
    the QAT fake-quant and PTQ calibration cannot drift apart on what "scale"
    means.
    """

    scale: float | torch.Tensor
    n_bits: int

    @property
    def per_channel(self) -> bool:
        return isinstance(self.scale, torch.Tensor) and self.scale.numel() > 1

    @property
    def is_power_of_two(self) -> bool:
        if isinstance(self.scale, torch.Tensor):
            frac = torch.log2(self.scale) % 1.0
            return bool(((frac < 1e-9) | (frac > 1 - 1e-9)).all())
        import math
        return self.scale > 0 and math.isclose(math.log2(self.scale) % 1.0, 0.0, abs_tol=1e-9)

    @property
    def exponent(self) -> int | torch.Tensor:
        """The shift amount a fixed-point datapath would use.

        Per-channel, this is one shift PER OUTPUT CHANNEL -- still a shift, not
        a multiply, which is why per-channel and dynamic fixed point go together
        rather than competing.
        """
        if isinstance(self.scale, torch.Tensor):
            return torch.round(torch.log2(self.scale)).to(torch.int32)
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


def _to_power_of_two(scale: float | torch.Tensor) -> float | torch.Tensor:
    """Round UP to the next power of two.

    Up, never to nearest: rounding down would make the scale too small for the
    observed maximum and clip it. Costs up to 2x of range, which is the price of
    turning the requantisation multiply into a shift.
    """
    if isinstance(scale, torch.Tensor):
        return torch.where(scale > 0, torch.pow(2.0, torch.ceil(torch.log2(scale))),
                           torch.ones_like(scale))
    import math
    return 2.0 ** math.ceil(math.log2(scale)) if scale > 0 else 1.0


def compute_symmetric_scale(
    x: torch.Tensor, n_bits: int = 8, power_of_two: bool = False, per_channel: bool = False
) -> QuantScale:
    """Symmetric scale for `x`, per tensor or per output channel.

    Per-channel matters most for DEPTHWISE convolutions, which have one
    independent filter per channel and no mixing to even them out. A single
    tensor-wide scale is then set by the loudest channel and leaves the quiet
    ones only a few quantisation levels -- the failure mode Krishnamoorthi
    (2018) measured on MobileNet, and this network is depthwise-separable
    throughout.
    """
    qmax = 2 ** (n_bits - 1) - 1
    if per_channel:
        # Reduce over everything except dim 0, keeping the shape broadcastable
        # against the weight tensor.
        flat = x.detach().flatten(1).abs().amax(dim=1)
        scale = torch.where(flat > 0, flat / qmax, torch.ones_like(flat))
        if power_of_two:
            scale = _to_power_of_two(scale)
        return QuantScale(scale=scale.reshape(-1, *([1] * (x.dim() - 1))), n_bits=n_bits)
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

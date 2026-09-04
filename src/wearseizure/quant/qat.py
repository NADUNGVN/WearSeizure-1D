"""Quantization-aware training via fake-quant (straight-through estimator) on
weights and activations -- INT8 W/A per memo Table 1. Wrapping happens by
replacing `nn.Conv1d`/`nn.Linear` submodules in-place, so the same
`WearSeizure1D`/baseline classes are reused unmodified for FP32, PTQ, and QAT.

`int_reference.py` replays the exact scales this module settles on in true
integer arithmetic -- the two are meant to be compared directly (memo 5.4).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from wearseizure.quant.scales import (
    QuantScale,
    compute_symmetric_scale,
    compute_symmetric_scale_from_max,
)


class FakeQuantSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: QuantScale) -> torch.Tensor:
        q = torch.clamp(torch.round(x / scale.scale), scale.qmin, scale.qmax)
        return q * scale.scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None  # straight-through: gradient passes unchanged


def fake_quantize(x: torch.Tensor, scale: QuantScale) -> torch.Tensor:
    return FakeQuantSTE.apply(x, scale)


class _QATMixin:
    """Shared activation-range tracking for QATConv1d / QATLinear.

    `calibrating` is distinct from `training`: PTQ calibrates (updates the
    running activation max) in `model.eval()` mode with no weight updates,
    while QAT updates it during normal training. Both freeze once neither
    flag is set (i.e. at inference / continuous test time).
    """

    def _init_qat_state(
        self, act_bits: int, weight_bits: int,
        weight_per_channel: bool = False, power_of_two: bool = False,
    ) -> None:
        self.act_bits = act_bits
        self.weight_bits = weight_bits
        # Both default off so every number already in the experiment log stays
        # reproducible: this is per-tensor with an arbitrary scale unless asked.
        self.weight_per_channel = weight_per_channel
        self.power_of_two = power_of_two
        self.calibrating = False
        self.momentum = 0.1
        self.register_buffer("act_running_max", torch.tensor(1e-4))

    def _quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.training or self.calibrating:
            with torch.no_grad():
                batch_max = x.detach().abs().max()
                self.act_running_max.mul_(1 - self.momentum).add_(self.momentum * batch_max)
        act_scale = compute_symmetric_scale_from_max(
            self.act_running_max, self.act_bits, power_of_two=self.power_of_two
        )
        return fake_quantize(x, act_scale)

    def _weight_scale(self, weight: torch.Tensor) -> QuantScale:
        return compute_symmetric_scale(
            weight, self.weight_bits,
            power_of_two=self.power_of_two, per_channel=self.weight_per_channel,
        )

    def last_scales(self, weight: torch.Tensor) -> tuple[QuantScale, QuantScale]:
        """(activation_scale, weight_scale) as of the most recent forward pass."""
        return (
            compute_symmetric_scale_from_max(
                self.act_running_max, self.act_bits, power_of_two=self.power_of_two
            ),
            self._weight_scale(weight),
        )


class QATConv1d(nn.Module, _QATMixin):
    def __init__(
        self, conv: nn.Conv1d, weight_bits: int = 8, act_bits: int = 8,
        weight_per_channel: bool = False, power_of_two: bool = False,
    ) -> None:
        super().__init__()
        self.conv = conv
        self._init_qat_state(act_bits, weight_bits, weight_per_channel, power_of_two)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = self._quantize_activation(x)
        weight_scale = self._weight_scale(self.conv.weight)
        w_q = fake_quantize(self.conv.weight, weight_scale)
        return F.conv1d(x_q, w_q, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)


class QATLinear(nn.Module, _QATMixin):
    def __init__(
        self, linear: nn.Linear, weight_bits: int = 8, act_bits: int = 8,
        weight_per_channel: bool = False, power_of_two: bool = False,
    ) -> None:
        super().__init__()
        self.linear = linear
        self._init_qat_state(act_bits, weight_bits, weight_per_channel, power_of_two)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = self._quantize_activation(x)
        weight_scale = self._weight_scale(self.linear.weight)
        w_q = fake_quantize(self.linear.weight, weight_scale)
        return F.linear(x_q, w_q, self.linear.bias)


def prepare_qat(
    model: nn.Module, weight_bits: int = 8, act_bits: int = 8,
    weight_per_channel: bool = False, power_of_two: bool = False,
) -> nn.Module:
    opts = (weight_bits, act_bits, weight_per_channel, power_of_two)
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Conv1d):
            setattr(model, name, QATConv1d(child, *opts))
        elif isinstance(child, nn.Linear):
            setattr(model, name, QATLinear(child, *opts))
        else:
            prepare_qat(child, *opts)
    return model


def set_calibrating(model: nn.Module, flag: bool) -> None:
    for module in model.modules():
        if isinstance(module, (QATConv1d, QATLinear)):
            module.calibrating = flag

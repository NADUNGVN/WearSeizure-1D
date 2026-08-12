"""Symmetric per-tensor quantization scale (memo Table 1: INT8 W/A, INT32
accumulator). Shared by QAT fake-quant, PTQ calibration, and the integer
reference so all three agree on what "scale" means for a given tensor.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class QuantScale:
    scale: float
    n_bits: int

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


def compute_symmetric_scale(x: torch.Tensor, n_bits: int = 8) -> QuantScale:
    qmax = 2 ** (n_bits - 1) - 1
    max_abs = float(x.detach().abs().max())
    scale = max_abs / qmax if max_abs > 0 else 1.0
    return QuantScale(scale=scale, n_bits=n_bits)


def compute_symmetric_scale_from_max(max_abs: torch.Tensor | float, n_bits: int = 8) -> QuantScale:
    qmax = 2 ** (n_bits - 1) - 1
    max_abs_f = float(max_abs)
    scale = max_abs_f / qmax if max_abs_f > 0 else 1.0
    return QuantScale(scale=scale, n_bits=n_bits)

"""Model-level metrics: params/MACs/bytes budget (Table 4/6) and FP32 vs
INT8 vs W4A8 quantization loss comparison, all evaluated on the same
split/seed per memo 7.1.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ModelSize:
    params: int
    macs: int
    weight_bytes: int
    activation_bytes_per_window: int


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_macs(model: torch.nn.Module, input_shape: tuple[int, ...]) -> int:
    """Requires `thop` (analysis extra). input_shape excludes the batch dim."""
    from thop import profile

    dummy = torch.zeros(1, *input_shape)
    macs, _ = profile(model, inputs=(dummy,), verbose=False)
    return int(macs)


def weight_bytes(model: torch.nn.Module, bits: int) -> int:
    n_params = count_params(model)
    return (n_params * bits + 7) // 8


def model_size(model: torch.nn.Module, input_shape: tuple[int, ...], weight_bits: int = 32) -> ModelSize:
    params = count_params(model)
    macs = count_macs(model, input_shape)
    return ModelSize(
        params=params,
        macs=macs,
        weight_bytes=weight_bytes(model, weight_bits),
        activation_bytes_per_window=0,  # filled in once QAT activation profiling exists (Gate G2)
    )


def quantization_loss_pp(fp32_metric: float, quantized_metric: float) -> float:
    """Percentage-point loss, e.g. event sensitivity FP32 0.99 vs INT8 0.985 -> 0.5pp."""
    return (fp32_metric - quantized_metric) * 100.0

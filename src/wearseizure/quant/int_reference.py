"""Pure NumPy integer reference for a single quantized Conv1d layer -- the
first link in the memo 5.4 golden chain (PyTorch FP32 -> QAT fake-quant ->
integer reference -> RTL simulation -> FPGA). It replays exactly the scales a
`QATConv1d` settled on, but in true int32 arithmetic with explicit
saturation, instead of PyTorch's float simulation of quantization.

Scoped to one layer at a time (extract weight/activation scale from a trained
`QATConv1d`, run both paths on the same input) rather than a full-model
reimplementation: the point at this stage is to prove the *scale/rounding
convention* is bit-exact-reproducible outside PyTorch, which is what the RTL
implementation will need to match. A full-model integer reimplementation is
Gate G3 work, once the RTL microarchitecture (memo 4.5) fixes the exact
per-layer requantization convention.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1


def quantize_int8(x: np.ndarray, scale: float) -> np.ndarray:
    q = np.round(x / scale)
    return np.clip(q, -128, 127).astype(np.int32)


def saturate_int32(x: np.ndarray) -> np.ndarray:
    return np.clip(x, INT32_MIN, INT32_MAX)


@dataclass(frozen=True)
class IntConv1dParams:
    weight_int8: np.ndarray  # (out_ch, in_ch/groups, k)
    bias_int32: np.ndarray | None
    weight_scale: float
    input_scale: float
    output_scale: float
    stride: int = 1
    padding: int = 0
    dilation: int = 1
    groups: int = 1


def int_conv1d_accumulate(x_int8: np.ndarray, params: IntConv1dParams) -> np.ndarray:
    """Raw INT32-saturated accumulator, in units of `weight_scale * input_scale`
    -- i.e. before requantization to `output_scale`. Exposed separately so
    tests can dequantize directly (`acc * weight_scale * input_scale`) and
    compare against the QAT float path without an intermediate int8 clip.
    """
    out_ch, in_ch_per_group, k = params.weight_int8.shape
    groups = params.groups
    padded = np.pad(x_int8, ((0, 0), (params.padding, params.padding))).astype(np.int64)
    L_padded = padded.shape[1]
    L_out = (L_padded - params.dilation * (k - 1) - 1) // params.stride + 1
    if L_out <= 0:
        raise ValueError(f"non-positive output length {L_out} for input length {x_int8.shape[1]}")

    out_per_group = out_ch // groups
    out = np.zeros((out_ch, L_out), dtype=np.int64)

    for g in range(groups):
        in_start, in_end = g * in_ch_per_group, (g + 1) * in_ch_per_group
        x_g = padded[in_start:in_end]
        for oc_local in range(out_per_group):
            oc = g * out_per_group + oc_local
            w = params.weight_int8[oc].astype(np.int64)  # (in_ch_per_group, k)
            acc = np.zeros(L_out, dtype=np.int64)
            for t in range(k):
                offset = t * params.dilation
                x_slice = x_g[:, offset : offset + params.stride * L_out : params.stride]
                acc += (w[:, t : t + 1] * x_slice).sum(axis=0)
            if params.bias_int32 is not None:
                acc += int(params.bias_int32[oc])
            out[oc] = saturate_int32(acc)

    return out


def int_conv1d(x_int8: np.ndarray, params: IntConv1dParams) -> np.ndarray:
    """Full path: accumulate, then requantize to `output_scale`, matching
    memo 4.5's "fused Conv-BN-ReLU-requant, accumulator INT32, saturation
    checked". Returns int8-range output (out_ch, L_out).
    """
    acc = int_conv1d_accumulate(x_int8, params)
    requant_scale = (params.weight_scale * params.input_scale) / params.output_scale
    return quantize_int8(acc.astype(np.float64) * requant_scale, scale=1.0)

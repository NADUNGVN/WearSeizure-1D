"""Compares the QAT fake-quant float path against the pure-NumPy integer
reference on the same trained weights/input (memo 5.4 golden chain, first
link: FP32 -> QAT fake-quant -> integer reference).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from wearseizure.quant.int_reference import IntConv1dParams, int_conv1d_accumulate, quantize_int8
from wearseizure.quant.qat import QATConv1d
from wearseizure.quant.scales import compute_symmetric_scale, compute_symmetric_scale_from_max


def test_int_reference_accumulator_matches_qat_float_path():
    torch.manual_seed(0)
    conv = nn.Conv1d(3, 5, kernel_size=3, stride=2, padding=1, dilation=1, groups=1, bias=True)
    qat_conv = QATConv1d(conv, weight_bits=8, act_bits=8)
    qat_conv.train()

    x = torch.randn(1, 3, 16)
    for _ in range(5):  # warm up the running activation max
        qat_conv(torch.randn(1, 3, 16))
    qat_out = qat_conv(x)

    act_scale = compute_symmetric_scale_from_max(qat_conv.act_running_max, 8)
    weight_scale = compute_symmetric_scale(conv.weight, 8)

    x_q = act_scale.quantize(x[0]).numpy().astype(np.int32)  # (3, 16)
    w_q = weight_scale.quantize(conv.weight.detach()).numpy().astype(np.int32)  # (5, 3, 3)
    bias_q = np.round(conv.bias.detach().numpy() / (weight_scale.scale * act_scale.scale)).astype(np.int64)

    params = IntConv1dParams(
        weight_int8=w_q, bias_int32=bias_q, weight_scale=weight_scale.scale, input_scale=act_scale.scale,
        output_scale=1.0, stride=2, padding=1, dilation=1, groups=1,
    )
    acc = int_conv1d_accumulate(x_q, params)
    dequantized = acc.astype(np.float64) * weight_scale.scale * act_scale.scale

    # The QAT path fake-quantizes x and w with the SAME scales before calling
    # F.conv1d in float; the integer path does the identical multiply-add in
    # true int64. Both must agree up to float32 rounding noise.
    np.testing.assert_allclose(dequantized, qat_out[0].detach().numpy(), atol=1e-3, rtol=1e-4)


def test_quantize_int8_clips_to_range():
    x = np.array([-1000.0, 0.0, 1000.0])
    q = quantize_int8(x, scale=1.0)
    assert q.tolist() == [-128, 0, 127]


def test_int_conv1d_accumulate_matches_manual_dot_product_single_output():
    # 1 in-channel, 1 out-channel, kernel=2, stride=1, no padding/dilation:
    # trivially checkable by hand.
    x_q = np.array([[1, 2, 3, 4]], dtype=np.int32)
    w_q = np.array([[[2, -1]]], dtype=np.int32)  # out_ch=1, in_ch=1, k=2
    params = IntConv1dParams(
        weight_int8=w_q, bias_int32=None, weight_scale=1.0, input_scale=1.0, output_scale=1.0,
        stride=1, padding=0, dilation=1, groups=1,
    )
    acc = int_conv1d_accumulate(x_q, params)
    # position0: 2*1 + (-1)*2 = 0 ; position1: 2*2 + (-1)*3 = 1 ; position2: 2*3 + (-1)*4 = 2
    np.testing.assert_array_equal(acc[0], [0, 1, 2])

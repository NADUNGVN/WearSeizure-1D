"""Building blocks for WearSeizure-1D (memo 4.4/4.5): depthwise-separable
convolution, and the multi-scale [DW k3 || DW k5,dilation] branch used by
stages B2-B4. Padding for the k5 branch is chosen so its output length always
equals the k3 branch's output length regardless of stride/dilation, which is
what makes the concatenation always shape-valid.
"""
from __future__ import annotations

import torch
from torch import nn


class DepthwiseSeparableConv1d(nn.Module):
    """Single-kernel depthwise-separable conv (used for the Stem->B1 path and
    the Context stage), fusing BN+ReLU after both the depthwise and pointwise
    stage per memo 4.5 ("fuse Conv-BN-ReLU-requant").
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1) -> None:
        super().__init__()
        padding = ((kernel_size - 1) * dilation) // 2
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class MultiScaleDilatedBlock(nn.Module):
    """[DW k3 || DW k5,dilation] -> concat -> pointwise conv (Table 4, B2-B4).

    The RTL tap generator (memo 4.5) is shared across k=3/5 and dilation
    1/2/4 and time-multiplexed on one PE bank -- this module is the software
    counterpart of that shared datapath, not an independent implementation
    per branch.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int, dilation: int) -> None:
        super().__init__()
        self.branch_k3 = nn.Conv1d(
            in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False
        )
        pad_k5 = ((5 - 1) * dilation) // 2
        self.branch_k5 = nn.Conv1d(
            in_channels, in_channels, kernel_size=5, stride=stride, padding=pad_k5,
            dilation=dilation, groups=in_channels, bias=False,
        )
        self.bn_dw = nn.BatchNorm1d(in_channels * 2)
        self.act_dw = nn.ReLU(inplace=True)
        self.pointwise = nn.Conv1d(in_channels * 2, out_channels, kernel_size=1, bias=False)
        self.bn_pw = nn.BatchNorm1d(out_channels)
        self.act_pw = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b3 = self.branch_k3(x)
        b5 = self.branch_k5(x)
        merged = torch.cat([b3, b5], dim=1)
        merged = self.act_dw(self.bn_dw(merged))
        out = self.pointwise(merged)
        out = self.bn_pw(out)
        return self.act_pw(out)

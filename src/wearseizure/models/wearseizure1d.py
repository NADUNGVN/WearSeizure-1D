"""WearSeizure-1D (Table 4 of the Research Decision Memo).

Stem (Conv1D k7,s2) -> B1 (single-branch DW k5,s2) -> B2/B3/B4 (multi-scale
[DW k3 || DW k5,dilation], s2) -> Context (2x dilated depthwise-separable
conv, stride 1) -> GAP -> FC(2 logits).

Default channel widths are the memo's starting point (target ~13,810 params /
0.644M MACs, hard ceiling 32k params / 2M MACs before Gate G1). Exact tuning
to hit the target happens during Gate G1b ablation (kernel/dilation
on-off, standard vs depthwise-separable, etc. -- memo 7.2); this
implementation only needs to respect the hard budget, which
tests/unit/test_models_shapes_and_budget.py checks.
"""
from __future__ import annotations

import torch
from torch import nn

from wearseizure.models.layers import DepthwiseSeparableConv1d, MultiScaleDilatedBlock


class WearSeizure1D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        input_len: int = 1024,
        stem_out_channels: int = 8,
        stage_out_channels: tuple[int, int, int, int] = (16, 24, 32, 48),
        context_channels: int = 64,
        dilations: tuple[int, int, int] = (1, 2, 4),
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.input_len = input_len
        s1, s2, s3, s4 = stage_out_channels
        d1, d2, d3 = dilations

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, stem_out_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(stem_out_channels),
            nn.ReLU(inplace=True),
        )
        self.b1 = DepthwiseSeparableConv1d(stem_out_channels, s1, kernel_size=5, stride=2, dilation=1)
        self.b2 = MultiScaleDilatedBlock(s1, s2, stride=2, dilation=d1)
        self.b3 = MultiScaleDilatedBlock(s2, s3, stride=2, dilation=d2)
        self.b4 = MultiScaleDilatedBlock(s3, s4, stride=2, dilation=d3)

        self.context = nn.Sequential(
            DepthwiseSeparableConv1d(s4, context_channels, kernel_size=5, stride=1, dilation=8),
            DepthwiseSeparableConv1d(context_channels, context_channels, kernel_size=5, stride=1, dilation=16),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(context_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_len:
            raise ValueError(f"expected input length {self.input_len}, got {x.shape[-1]}")
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.context(x)
        x = self.gap(x).flatten(1)
        return self.classifier(x)

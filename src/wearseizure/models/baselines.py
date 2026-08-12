"""Baselines that must be reproduced under the new leakage-safe protocol
before WearSeizure-1D's benefit can be claimed (memo 7.1 #1-#2): the
Frontiers 2024 architecture, and a compact standard (non-depthwise) 1D-CNN
sized like the ASICON 2021 baseline. Both exist to separate "the new
protocol changed the numbers" from "the new model changed the numbers".
"""
from __future__ import annotations

import torch
from torch import nn


class FrontiersBaseline2D(nn.Module):
    """Reproduction of the Frontiers 2024 two-branch stacked CNN (kernels
    1x3 and 1x5). Implemented with Conv2d over a (1, input_len) "image" so the
    kernel notation matches the source paper directly; functionally this is
    still a 1D signal processed by two parallel branches.
    """

    def __init__(
        self,
        in_channels: int = 1,
        input_len: int = 1024,
        branch_kernels: tuple[tuple[int, int], ...] = ((1, 3), (1, 5)),
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.input_len = input_len
        self.branches = nn.ModuleList()
        for kh, kw in branch_kernels:
            pad = (kh // 2, kw // 2)
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, 16, kernel_size=(kh, kw), padding=pad, bias=False),
                    nn.BatchNorm2d(16),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(16, 32, kernel_size=(kh, kw), stride=(1, 2), padding=pad, bias=False),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                )
            )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(32 * len(branch_kernels), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2d = x.unsqueeze(2)  # (B, C, 1, input_len)
        feats = [self.pool(branch(x2d)).flatten(1) for branch in self.branches]
        return self.classifier(torch.cat(feats, dim=1))


class Compact1DBaseline(nn.Module):
    """Standard (non-depthwise-separable) 1D-CNN, ASICON-2021-scale
    (config target ~7k params). Channel widths are a starting point; exact
    tuning towards `configs/model/baseline_compact1d_7k.yaml`'s param_target
    happens during Gate G1a baseline reproduction, not hardcoded here.
    """

    def __init__(self, in_channels: int = 1, input_len: int = 1024, num_classes: int = 2) -> None:
        super().__init__()
        self.input_len = input_len
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(48, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))

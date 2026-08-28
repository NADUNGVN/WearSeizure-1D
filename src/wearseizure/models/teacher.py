"""Multi-channel teacher for lever L3.

Deliberately NOT budget-constrained. Every other model in this repository is
sized against the Gate G1 ceiling (32k params / 2M MACs) because it has to fit
on a Zynq-7020. This one never runs on hardware and never appears in a reported
result -- it exists only to produce soft targets during training, so the design
goal is simply "learn the easier 18-channel problem well".

It is still small by any normal standard: the fold it trains on is one
patient's recordings, so a large model would overfit long before it helped.
"""
from __future__ import annotations

import torch
from torch import nn


class MultiChannelTeacher(nn.Module):
    """Standard 1D CNN over `(B, C, L)`.

    Cross-channel mixing happens in the very first convolution, which is the
    whole point: that is the information the single-channel student cannot
    reconstruct, and the only thing the teacher has that the student does not.
    """

    def __init__(
        self,
        in_channels: int,
        widths: tuple[int, int, int, int] = (32, 64, 96, 128),
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        w1, w2, w3, w4 = widths
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, w1, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(w1),
            nn.ReLU(inplace=True),
            nn.Conv1d(w1, w2, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(w2),
            nn.ReLU(inplace=True),
            nn.Conv1d(w2, w3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(w3),
            nn.ReLU(inplace=True),
            nn.Conv1d(w3, w4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(w4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(w4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected (B, C, L), got shape {tuple(x.shape)}")
        return self.classifier(self.dropout(self.features(x).flatten(1)))


def build_teacher(in_channels: int) -> MultiChannelTeacher:
    """Factory matching `training/distill.fold_teacher_logits`'s expectation of
    a one-argument callable, so the channel count comes from the data rather
    than from a config that could disagree with it."""
    return MultiChannelTeacher(in_channels=in_channels)

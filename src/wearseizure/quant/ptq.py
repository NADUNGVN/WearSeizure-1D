"""Post-training quantization: wrap an already-trained FP32 model with the
same fake-quant modules QAT uses, calibrate activation ranges on a handful of
batches (weights are not further trained), then freeze. Used as the PTQ
comparison point against QAT in the FP32/PTQ/QAT ablation (memo 7.1 #3).
"""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from wearseizure.quant.qat import prepare_qat, set_calibrating


@torch.no_grad()
def calibrate(model: nn.Module, calibration_loader: DataLoader, n_batches: int = 20, device: str = "cpu") -> None:
    model.eval()
    model.to(device)
    set_calibrating(model, True)
    for i, (x, _y) in enumerate(calibration_loader):
        model(x.to(device))
        if i + 1 >= n_batches:
            break
    set_calibrating(model, False)


def prepare_ptq(
    model: nn.Module,
    calibration_loader: DataLoader,
    weight_bits: int = 8,
    act_bits: int = 8,
    device: str = "cpu",
) -> nn.Module:
    model = prepare_qat(model, weight_bits=weight_bits, act_bits=act_bits)
    calibrate(model, calibration_loader, device=device)
    return model

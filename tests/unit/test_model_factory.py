"""Regression test for the window/model input_len desync bug: switching
`window=w2s_stride1s` (memo 7.2 window ablation) used to crash deep inside
training with "expected input length 1024, got 512" because model configs
hardcoded input_len=1024 independently of window_s. models/factory.py now
derives input_len from window_s * fs_hz instead.
"""
from __future__ import annotations

from omegaconf import OmegaConf

from wearseizure.models.baselines import Compact1DBaseline, FrontiersBaseline2D
from wearseizure.models.factory import build_model
from wearseizure.models.wearseizure1d import WearSeizure1D


def _cfg(model_name: str, window_s: float, fs_hz: float = 256.0):
    return OmegaConf.create(
        {
            "data": {"fs_hz": fs_hz},
            "window": {"window_s": window_s, "stride_s": 1.0},
            "model": {
                "name": model_name,
                "in_channels": 1,
                "input_len": 1024,  # intentionally wrong/stale to prove it's ignored
                "stem_out_channels": 8,
                "stage_out_channels": [16, 24, 32, 48],
                "context_channels": 64,
                "dilations": [1, 2, 4],
                "kernel_mode": "multi_scale",
                "num_classes": 2,
                "branch_kernels": [[1, 3], [1, 5]],
            },
        }
    )


def test_build_model_derives_input_len_from_window_s_not_config():
    model = build_model(_cfg("wearseizure1d", window_s=2.0))
    assert isinstance(model, WearSeizure1D)
    assert model.input_len == 512  # 2s * 256Hz, NOT the stale 1024 in cfg.model.input_len


def test_build_model_matches_default_4s_window():
    model = build_model(_cfg("wearseizure1d", window_s=4.0))
    assert model.input_len == 1024


def test_build_model_derives_input_len_for_baselines_too():
    frontiers = build_model(_cfg("baseline_frontiers2d", window_s=2.0))
    compact = build_model(_cfg("baseline_compact1d_7k", window_s=2.0))
    assert isinstance(frontiers, FrontiersBaseline2D)
    assert isinstance(compact, Compact1DBaseline)
    assert frontiers.input_len == 512
    assert compact.input_len == 512

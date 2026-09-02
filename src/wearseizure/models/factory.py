"""Model factory shared by scripts/train.py and scripts/rethreshold.py, so
adding or changing a model config only requires updating this one place.
"""
from __future__ import annotations

from wearseizure.models.baselines import Compact1DBaseline, FrontiersBaseline2D
from wearseizure.models.wearseizure1d import WearSeizure1D


def _input_len(cfg) -> int:
    """Samples per window, derived from window_s * fs_hz rather than read
    from a hardcoded `model.input_len` in each model config. Those configs
    were written assuming the default 4s window (1024 samples @ 256Hz); a
    hardcoded value silently went stale the first time a window ablation
    (memo 7.2: window=w2s_stride1s) was tried, since nothing kept the two
    config groups in sync -- the model raised "expected input length 1024,
    got 512" deep inside training. Deriving it here means switching
    `window=...` can never desync the model's expected input length again.
    """
    return round(cfg.window.window_s * cfg.data.fs_hz)


def _build_wearseizure1d(cfg):
    return WearSeizure1D(
        in_channels=cfg.model.in_channels,
        input_len=_input_len(cfg),
        stem_out_channels=cfg.model.stem_out_channels,
        stage_out_channels=tuple(cfg.model.stage_out_channels),
        context_channels=cfg.model.context_channels,
        dilations=tuple(cfg.model.dilations),
        num_classes=cfg.model.num_classes,
        kernel_mode=cfg.model.get("kernel_mode", "multi_scale"),
    )


MODEL_FACTORIES = {
    # All four share one builder -- only kernel_mode/dilations differ between
    # configs/model/wearseizure1d*.yaml (memo 7.2 kernel/dilation ablation).
    "wearseizure1d": _build_wearseizure1d,
    "wearseizure1d_k3only": _build_wearseizure1d,
    "wearseizure1d_k5only": _build_wearseizure1d,
    # The capacity ladder: same builder, only context/stage widths differ.
    "wearseizure1d_k5only_ctx16": _build_wearseizure1d,
    "wearseizure1d_k5only_wide": _build_wearseizure1d,
    "wearseizure1d_nodilation": _build_wearseizure1d,
    "baseline_frontiers2d": lambda cfg: FrontiersBaseline2D(
        in_channels=cfg.model.in_channels,
        input_len=_input_len(cfg),
        branch_kernels=tuple(tuple(k) for k in cfg.model.branch_kernels),
        num_classes=cfg.model.num_classes,
    ),
    "baseline_compact1d_7k": lambda cfg: Compact1DBaseline(
        in_channels=cfg.model.in_channels, input_len=_input_len(cfg), num_classes=cfg.model.num_classes
    ),
}


def build_model(cfg):
    factory = MODEL_FACTORIES.get(cfg.model.name)
    if factory is None:
        raise ValueError(f"unknown model {cfg.model.name!r}, expected one of {list(MODEL_FACTORIES)}")
    return factory(cfg)

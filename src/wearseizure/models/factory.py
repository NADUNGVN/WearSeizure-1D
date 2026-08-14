"""Model factory shared by scripts/train.py and scripts/rethreshold.py, so
adding or changing a model config only requires updating this one place.
"""
from __future__ import annotations

from wearseizure.models.baselines import Compact1DBaseline, FrontiersBaseline2D
from wearseizure.models.wearseizure1d import WearSeizure1D

MODEL_FACTORIES = {
    "wearseizure1d": lambda cfg: WearSeizure1D(
        in_channels=cfg.model.in_channels,
        input_len=cfg.model.input_len,
        stem_out_channels=cfg.model.stem_out_channels,
        stage_out_channels=tuple(cfg.model.stage_out_channels),
        context_channels=cfg.model.context_channels,
        dilations=tuple(cfg.model.dilations),
        num_classes=cfg.model.num_classes,
    ),
    "baseline_frontiers2d": lambda cfg: FrontiersBaseline2D(
        in_channels=cfg.model.in_channels,
        input_len=cfg.model.input_len,
        branch_kernels=tuple(tuple(k) for k in cfg.model.branch_kernels),
        num_classes=cfg.model.num_classes,
    ),
    "baseline_compact1d_7k": lambda cfg: Compact1DBaseline(
        in_channels=cfg.model.in_channels, input_len=cfg.model.input_len, num_classes=cfg.model.num_classes
    ),
}


def build_model(cfg):
    factory = MODEL_FACTORIES.get(cfg.model.name)
    if factory is None:
        raise ValueError(f"unknown model {cfg.model.name!r}, expected one of {list(MODEL_FACTORIES)}")
    return factory(cfg)

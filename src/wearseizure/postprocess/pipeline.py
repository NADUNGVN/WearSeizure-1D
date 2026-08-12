"""Dispatch to the configured postprocess method (memo 7.2 ablation: raw
threshold -> EMA -> full hysteresis+run-length are meant to be compared
head-to-head via `configs/postprocess/*.yaml`).
"""
from __future__ import annotations

import numpy as np

from wearseizure.postprocess.ema import ema_smooth
from wearseizure.postprocess.hysteresis import (
    PostprocessParams,
    hysteresis_alarms,
    raw_threshold_alarms,
)


def run_postprocess(
    end_sec: np.ndarray, scores: np.ndarray, params: PostprocessParams
) -> list[tuple[float, float]]:
    if params.method == "raw_threshold":
        return raw_threshold_alarms(end_sec, scores, params)
    if params.method == "ema":
        smoothed = ema_smooth(scores, params.ema_alpha)
        return raw_threshold_alarms(end_sec, smoothed, params)
    if params.method == "hysteresis_runlength":
        smoothed = ema_smooth(scores, params.ema_alpha)
        return hysteresis_alarms(end_sec, smoothed, params)
    raise ValueError(f"unknown postprocess method {params.method!r}")

from __future__ import annotations

import numpy as np
import pytest

from wearseizure.postprocess.ema import ema_smooth
from wearseizure.postprocess.hysteresis import (
    PostprocessParams,
    hysteresis_alarms,
    merge_intervals,
    raw_threshold_alarms,
)
from wearseizure.postprocess.pipeline import run_postprocess


def test_ema_smooth_first_value_passthrough():
    scores = np.array([0.5, 1.0, 1.0, 1.0])
    smoothed = ema_smooth(scores, alpha=0.5)
    assert smoothed[0] == 0.5
    assert smoothed[-1] > smoothed[0]


def test_ema_smooth_rejects_invalid_alpha():
    with pytest.raises(ValueError):
        ema_smooth(np.array([0.1, 0.2]), alpha=0.0)


def test_merge_intervals_joins_close_alarms_but_not_far_ones():
    merged = merge_intervals([(0.0, 1.0), (1.5, 2.0), (10.0, 11.0)], gap_s=1.0)
    assert merged == [(0.0, 2.0), (10.0, 11.0)]


def test_raw_threshold_alarms_basic():
    # Alarm start/end are the end_sec timestamps of the windows where the
    # score crosses above/below threshold -- here scores go above at t=2.0
    # (index 1) and back below at t=4.0 (index 3).
    end_sec = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    scores = np.array([0.1, 0.9, 0.9, 0.1, 0.1])
    params = PostprocessParams(method="raw_threshold", threshold=0.5)
    alarms = raw_threshold_alarms(end_sec, scores, params)
    assert alarms == [(2.0, 4.0)]


def test_hysteresis_requires_off_below_on():
    params = PostprocessParams(method="hysteresis_runlength", threshold_on=0.5, threshold_off=0.6)
    with pytest.raises(ValueError):
        hysteresis_alarms(np.array([1.0]), np.array([0.7]), params)


def test_hysteresis_run_length_suppresses_brief_spikes():
    end_sec = np.arange(1, 11, dtype=float)
    scores = np.array([0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])  # single-sample spike
    params = PostprocessParams(
        method="hysteresis_runlength", threshold_on=0.5, threshold_off=0.3, run_length=3, event_merge_gap_s=0.0
    )
    alarms = hysteresis_alarms(end_sec, scores, params)
    assert alarms == [], "a single sample above threshold_on must not trigger with run_length=3"


def test_hysteresis_run_length_fires_on_sustained_rise():
    end_sec = np.arange(1, 11, dtype=float)
    scores = np.array([0.1, 0.6, 0.7, 0.8, 0.8, 0.2, 0.1, 0.1, 0.1, 0.1])
    params = PostprocessParams(
        method="hysteresis_runlength", threshold_on=0.5, threshold_off=0.3, run_length=2, event_merge_gap_s=0.0
    )
    alarms = hysteresis_alarms(end_sec, scores, params)
    assert len(alarms) == 1
    start, end = alarms[0]
    assert start == 3.0  # second consecutive sample >= threshold_on
    assert end == 6.0  # first sample dropping below threshold_off


def test_run_postprocess_dispatches_by_method():
    end_sec = np.arange(1, 6, dtype=float)
    scores = np.array([0.1, 0.9, 0.9, 0.1, 0.1])
    raw = run_postprocess(end_sec, scores, PostprocessParams(method="raw_threshold", threshold=0.5))
    assert raw == [(2.0, 4.0)]

    with pytest.raises(ValueError):
        run_postprocess(end_sec, scores, PostprocessParams(method="not_a_method"))

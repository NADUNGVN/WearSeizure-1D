"""Alarm generation from a (smoothed) score stream: raw threshold, and full
two-threshold hysteresis + run-length + event-merge (memo 4.5).

`threshold`/`threshold_on`/`threshold_off` in `PostprocessParams` are only
ever meant to be populated by `training.threshold_selection` (fit on
validation, then frozen) -- nothing in this module fits anything.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PostprocessParams:
    method: str
    ema_alpha: float = 0.0
    threshold: float = 0.5  # raw_threshold / ema methods
    threshold_on: float = 0.5  # hysteresis_runlength
    threshold_off: float = 0.5
    run_length: int = 1
    event_merge_gap_s: float = 0.0


def merge_intervals(intervals: list[tuple[float, float]], gap_s: float) -> list[tuple[float, float]]:
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start - merged[-1][1] <= gap_s:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(m[0], m[1]) for m in merged]


def raw_threshold_alarms(
    end_sec: np.ndarray, scores: np.ndarray, params: PostprocessParams
) -> list[tuple[float, float]]:
    above = scores >= params.threshold
    alarms: list[tuple[float, float]] = []
    start = None
    for t, a in zip(end_sec, above):
        if a and start is None:
            start = float(t)
        elif not a and start is not None:
            alarms.append((start, float(t)))
            start = None
    if start is not None:
        alarms.append((start, float(end_sec[-1])))
    return merge_intervals(alarms, params.event_merge_gap_s)


def hysteresis_alarms(
    end_sec: np.ndarray, scores: np.ndarray, params: PostprocessParams
) -> list[tuple[float, float]]:
    if params.threshold_off >= params.threshold_on:
        raise ValueError(
            f"threshold_off ({params.threshold_off}) must be < threshold_on "
            f"({params.threshold_on}) for hysteresis to have an effect"
        )
    state_on = False
    run_count = 0
    alarm_start: float | None = None
    alarms: list[tuple[float, float]] = []

    for t, s in zip(end_sec, scores):
        if not state_on:
            if s >= params.threshold_on:
                run_count += 1
                if run_count >= params.run_length:
                    state_on = True
                    alarm_start = float(t)
            else:
                run_count = 0
        else:
            if s < params.threshold_off:
                alarms.append((alarm_start, float(t)))
                state_on = False
                run_count = 0
                alarm_start = None

    if state_on and alarm_start is not None:
        alarms.append((alarm_start, float(end_sec[-1])))

    return merge_intervals(alarms, params.event_merge_gap_s)

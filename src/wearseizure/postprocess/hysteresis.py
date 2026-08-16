"""Alarm generation from a (smoothed) score stream: raw threshold, and full
two-threshold hysteresis + run-length + event-merge (memo 4.5).

`threshold`/`threshold_on`/`threshold_off` in `PostprocessParams` are only
ever meant to be populated by `training.threshold_selection` (fit on
validation, then frozen) -- nothing in this module fits anything.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ALARM_TIMESTAMP_FRACTIONS = {
    # fraction of window_s subtracted from the window's end_sec decision point
    "window_end": 0.0,    # causal default: the alarm is stamped when the decision was actually made
    "window_center": 0.5,
    "window_start": 1.0,  # comparability only -- credits the alarm to signal the model had not finished reading
}


@dataclass(frozen=True)
class PostprocessParams:
    method: str
    ema_alpha: float = 0.0
    threshold: float = 0.5  # raw_threshold / ema methods
    threshold_on: float = 0.5  # hysteresis_runlength
    threshold_off: float = 0.5
    run_length: int = 1
    event_merge_gap_s: float = 0.0
    # Alarm timestamp convention (see docs/RESEARCH_REALITY_CHECK.md section 2).
    # `window_end` reproduces every result recorded in docs/EXPERIMENT_LOG_G1a.md
    # and is the only causally honest choice: the decision genuinely cannot be
    # made before the window it is computed from has been read. The other two
    # exist because published single-channel baselines report latencies *below*
    # their own window length (Chung et al. 2024: 3.3s mean with a 4s window),
    # which is only possible under an earlier crediting convention -- so a
    # like-for-like comparison needs the option to be stated explicitly rather
    # than assumed. `window_s` defaults to 0.0, which makes the convention a
    # no-op unless the caller deliberately supplies the window length.
    alarm_timestamp: str = "window_end"
    window_s: float = 0.0


def alarm_timestamp_offset_s(params: PostprocessParams) -> float:
    """Seconds subtracted from each window's `end_sec` to place an alarm."""
    try:
        fraction = ALARM_TIMESTAMP_FRACTIONS[params.alarm_timestamp]
    except KeyError:
        raise ValueError(
            f"unknown alarm_timestamp {params.alarm_timestamp!r}, "
            f"expected one of {sorted(ALARM_TIMESTAMP_FRACTIONS)}"
        ) from None
    return fraction * params.window_s


def _stamp(t: float, offset_s: float) -> float:
    """Apply the alarm-timestamp convention, never going negative."""
    return max(0.0, float(t) - offset_s)


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
    offset = alarm_timestamp_offset_s(params)
    above = scores >= params.threshold
    alarms: list[tuple[float, float]] = []
    start = None
    for t, a in zip(end_sec, above):
        if a and start is None:
            start = _stamp(t, offset)
        elif not a and start is not None:
            alarms.append((start, _stamp(t, offset)))
            start = None
    if start is not None:
        alarms.append((start, _stamp(end_sec[-1], offset)))
    return merge_intervals(alarms, params.event_merge_gap_s)


def hysteresis_alarms(
    end_sec: np.ndarray, scores: np.ndarray, params: PostprocessParams
) -> list[tuple[float, float]]:
    if params.threshold_off >= params.threshold_on:
        raise ValueError(
            f"threshold_off ({params.threshold_off}) must be < threshold_on "
            f"({params.threshold_on}) for hysteresis to have an effect"
        )
    offset = alarm_timestamp_offset_s(params)
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
                    alarm_start = _stamp(t, offset)
            else:
                run_count = 0
        else:
            if s < params.threshold_off:
                alarms.append((alarm_start, _stamp(t, offset)))
                state_on = False
                run_count = 0
                alarm_start = None

    if state_on and alarm_start is not None:
        alarms.append((alarm_start, _stamp(end_sec[-1], offset)))

    return merge_intervals(alarms, params.event_merge_gap_s)

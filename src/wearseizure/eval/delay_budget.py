"""The protocol-imposed floor on detection delay, and the decomposition of a
measured delay into "floor" and "model reaction".

Why this module exists
----------------------
Across the 19 real-data runs in `docs/EXPERIMENT_LOG_G1a.md`, the
`detection_delay_mean_s` gate (<= 5.0s) was never cleared. It could not have
been: the measurement configuration alone puts a hard lower bound on any delay
this pipeline can report, and with the shipped defaults that bound is 13.0s.

An alarm is stamped at the `end_sec` of the window that satisfied the
run-length condition (`postprocess/hysteresis.py`), on a score stream already
smoothed by an EMA (`postprocess/pipeline.py`). So even a perfect classifier
that fires on the very first window touching a seizure cannot report a delay
below:

    floor = window_s + (run_length - 1) * stride_s + ((1 - alpha)/alpha) * stride_s
            \\_ window _/  \\_____ run-length _____/  \\______ EMA group delay ______/

with the window term reduced by whatever the alarm-timestamp convention credits
back (see `postprocess.hysteresis.ALARM_TIMESTAMP_FRACTIONS`).

The EMA term is the group delay of a first-order exponential filter
`y_t = a*x_t + (1-a)*y_{t-1}`, whose mean lag is `(1-a)/a` samples.

Reporting this alongside every delay number is what stops a configuration
artefact from being read as model weakness -- and it is what makes the
"detection_delay_floor_s" gate in `configs/eval/gates_v2_proposed.yaml`
checkable.
"""
from __future__ import annotations

from dataclasses import dataclass

from wearseizure.postprocess.hysteresis import ALARM_TIMESTAMP_FRACTIONS


@dataclass(frozen=True)
class DelayBudget:
    window_s: float
    stride_s: float
    run_length: int
    ema_alpha: float
    alarm_timestamp: str
    window_term_s: float
    run_length_term_s: float
    ema_term_s: float
    floor_s: float

    def to_dict(self) -> dict:
        return {
            "window_s": self.window_s,
            "stride_s": self.stride_s,
            "run_length": self.run_length,
            "ema_alpha": self.ema_alpha,
            "alarm_timestamp": self.alarm_timestamp,
            "window_term_s": self.window_term_s,
            "run_length_term_s": self.run_length_term_s,
            "ema_term_s": self.ema_term_s,
            "floor_s": self.floor_s,
        }


def ema_group_delay_samples(ema_alpha: float) -> float:
    """Mean lag, in samples, of `y_t = a*x_t + (1-a)*y_{t-1}`.

    `alpha <= 0` means no smoothing is applied (the `raw_threshold` method), so
    the lag is zero rather than infinite.
    """
    if ema_alpha <= 0.0:
        return 0.0
    if ema_alpha > 1.0:
        raise ValueError(f"ema_alpha must be in (0, 1], got {ema_alpha}")
    return (1.0 - ema_alpha) / ema_alpha


def delay_budget(
    window_s: float,
    stride_s: float,
    run_length: int = 1,
    ema_alpha: float = 0.0,
    alarm_timestamp: str = "window_end",
) -> DelayBudget:
    if window_s <= 0 or stride_s <= 0:
        raise ValueError(f"window_s and stride_s must be positive, got {window_s}, {stride_s}")
    if run_length < 1:
        raise ValueError(f"run_length must be >= 1, got {run_length}")
    try:
        credited_back = ALARM_TIMESTAMP_FRACTIONS[alarm_timestamp] * window_s
    except KeyError:
        raise ValueError(
            f"unknown alarm_timestamp {alarm_timestamp!r}, "
            f"expected one of {sorted(ALARM_TIMESTAMP_FRACTIONS)}"
        ) from None

    window_term = window_s - credited_back
    run_length_term = (run_length - 1) * stride_s
    ema_term = ema_group_delay_samples(ema_alpha) * stride_s

    return DelayBudget(
        window_s=window_s,
        stride_s=stride_s,
        run_length=run_length,
        ema_alpha=ema_alpha,
        alarm_timestamp=alarm_timestamp,
        window_term_s=window_term,
        run_length_term_s=run_length_term,
        ema_term_s=ema_term,
        floor_s=max(0.0, window_term + run_length_term + ema_term),
    )


def model_reaction_s(measured_delay_s: float, budget: DelayBudget) -> float:
    """The part of a measured delay that is actually attributable to the model.

    Clamped at zero: a measured delay below the floor means the alarm overlapped
    the event from before its onset (delays are clipped at 0 in
    `metrics_event.compute_event_metrics`), not that the model reacted early.
    """
    return max(0.0, measured_delay_s - budget.floor_s)

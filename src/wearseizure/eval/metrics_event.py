"""Event-level metrics (memo 5.2): the primary evidence for this project.
Segment accuracy is deliberately not computed here -- see metrics_segment.py
for the secondary, explicitly-labeled-as-secondary segment metrics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wearseizure.eval.event_matching import match_events_to_alarms


@dataclass(frozen=True)
class EventMetrics:
    n_events: int
    n_matched: int
    n_missed: int
    n_false_alarms: int
    sensitivity: float
    far_per_hour: float
    delays_s: list[float]
    exposure_hours: float


def compute_event_metrics(
    events: list[tuple[str, float, float]],
    alarms: list[tuple[float, float]],
    exposure_hours: float,
) -> EventMetrics:
    result = match_events_to_alarms(events, alarms)
    n_events = len(events)
    n_matched = len(result.matched)
    sensitivity = n_matched / n_events if n_events > 0 else float("nan")
    far_per_hour = len(result.false_alarms) / exposure_hours if exposure_hours > 0 else float("nan")
    # Delay is clipped at 0: an alarm overlapping an event but starting before
    # its onset is not a "negative delay" detection, it is an early/lucky
    # overlap and is reported as zero delay rather than a negative number.
    delays = [max(0.0, alarm[0] - event_interval[0]) for _, event_interval, alarm in result.matched]
    return EventMetrics(
        n_events=n_events,
        n_matched=n_matched,
        n_missed=len(result.missed_event_ids),
        n_false_alarms=len(result.false_alarms),
        sensitivity=sensitivity,
        far_per_hour=far_per_hour,
        delays_s=delays,
        exposure_hours=exposure_hours,
    )


def macro_mean(per_patient: dict[str, EventMetrics]) -> dict:
    sens = [m.sensitivity for m in per_patient.values() if not np.isnan(m.sensitivity)]
    far = [m.far_per_hour for m in per_patient.values() if not np.isnan(m.far_per_hour)]
    return {
        "sensitivity_macro": float(np.mean(sens)) if sens else float("nan"),
        "far_per_hour_macro": float(np.mean(far)) if far else float("nan"),
        "n_patients": len(per_patient),
    }


def micro_pooled(per_patient: dict[str, EventMetrics]) -> dict:
    total_events = sum(m.n_events for m in per_patient.values())
    total_matched = sum(m.n_matched for m in per_patient.values())
    total_false_alarms = sum(m.n_false_alarms for m in per_patient.values())
    total_exposure = sum(m.exposure_hours for m in per_patient.values())
    return {
        "sensitivity_micro": total_matched / total_events if total_events else float("nan"),
        "far_per_hour_micro": total_false_alarms / total_exposure if total_exposure else float("nan"),
        "n_events": total_events,
        "n_matched": total_matched,
        "n_false_alarms": total_false_alarms,
        "exposure_hours": total_exposure,
    }


def delay_stats(per_patient: dict[str, EventMetrics]) -> dict:
    all_delays = [d for m in per_patient.values() for d in m.delays_s]
    if not all_delays:
        return {"mean_s": float("nan"), "median_s": float("nan"), "p95_s": float("nan")}
    arr = np.asarray(all_delays)
    return {
        "mean_s": float(arr.mean()),
        "median_s": float(np.median(arr)),
        "p95_s": float(np.percentile(arr, 95)),
    }


def worst_patient(per_patient: dict[str, EventMetrics], min_events: int | None = None) -> dict:
    """Worst-performing patient on each axis.

    `min_events` implements the small-sample rule of
    `configs/eval/gates_v2_proposed.yaml`. Without it, the worst-sensitivity
    slot is permanently occupied by whichever patient has the fewest seizures:
    with n events a patient's sensitivity can only take the values k/n, so on
    CHB-MIT's 3-seizure cases (chb02, chb17) a >=0.85 gate silently means "must
    not miss a single seizure" and the reported number says nothing about the
    other twelve patients.

    When `min_events` is given, the `*_gated` keys report the worst patient
    among those with at least that many seizures -- the number the gate is
    actually meant to constrain -- while the ungated keys keep reporting the
    raw cohort worst so the small-sample patients are never hidden.
    """
    sens_values = {pid: m.sensitivity for pid, m in per_patient.items() if not np.isnan(m.sensitivity)}
    far_values = {pid: m.far_per_hour for pid, m in per_patient.items() if not np.isnan(m.far_per_hour)}
    worst_sens_patient = min(sens_values, key=sens_values.get) if sens_values else None
    worst_far_patient = max(far_values, key=far_values.get) if far_values else None

    if min_events is None:
        eligible = dict(sens_values)
        below_floor: list[str] = []
    else:
        eligible = {
            pid: s for pid, s in sens_values.items() if per_patient[pid].n_events >= min_events
        }
        below_floor = sorted(pid for pid in sens_values if pid not in eligible)
    gated_patient = min(eligible, key=eligible.get) if eligible else None

    return {
        "sensitivity": sens_values.get(worst_sens_patient, float("nan")),
        "sensitivity_patient": worst_sens_patient,
        # Seizure count of the patient the sensitivity gate lands on -- the
        # count has to travel with the value for the gate to be interpretable;
        # see `min_events_to_gate` in configs/eval/gates_v2_proposed.yaml.
        "sensitivity_patient_n_events": (
            per_patient[worst_sens_patient].n_events if worst_sens_patient is not None else 0
        ),
        "min_events_to_gate": min_events,
        "sensitivity_gated": eligible.get(gated_patient, float("nan")),
        "sensitivity_gated_patient": gated_patient,
        "sensitivity_gated_patient_n_events": (
            per_patient[gated_patient].n_events if gated_patient is not None else 0
        ),
        "patients_below_event_floor": below_floor,
        "far_per_hour": far_values.get(worst_far_patient, float("nan")),
        "far_per_hour_patient": worst_far_patient,
    }

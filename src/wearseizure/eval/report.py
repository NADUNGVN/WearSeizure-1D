"""Aggregate per-patient event metrics into the macro/micro/CI report and
check it against Table 6 gates (only enforced when profile.enforce_gates).
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from wearseizure.eval.bootstrap import clopper_pearson_ci, cluster_bootstrap_ci, poisson_rate_ci
from wearseizure.eval.delay_budget import DelayBudget, model_reaction_s
from wearseizure.eval.metrics_event import (
    EventMetrics,
    delay_stats,
    macro_mean,
    micro_pooled,
    worst_patient,
)

GATE_LEVELS = ("stretch", "target", "minimum")

# Keys in a gates file that are metadata blocks, not threshold tables -- see
# configs/eval/gates_v2_proposed.yaml. `_gate_level` must never run against them.
NON_GATE_KEYS = frozenset({"reproduction", "zero_shot_loso"})


def load_gates(gates_path: str) -> dict:
    return OmegaConf.to_container(OmegaConf.load(gates_path), resolve=True)


def _gate_level(value: float, thresholds: dict) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "unknown"
    direction = thresholds.get("direction", "higher_is_better")
    better = (lambda v, t: v <= t) if direction == "lower_is_better" else (lambda v, t: v >= t)
    for level in GATE_LEVELS:
        if level in thresholds and better(value, thresholds[level]):
            return level
    return "below_minimum"


def _small_sample_exempt(key: str, thresholds: dict, flat_metrics: dict[str, float]) -> bool:
    """True when a gate declares `min_events_to_gate` and the patient it lands
    on has fewer events than that.

    Only `worst_patient_sensitivity` uses this today. The rule exists because a
    sensitivity threshold is not expressible on a patient with very few
    seizures: with 3 events the only reachable values are 0, 1/3, 2/3 and 1, so
    a >=0.85 gate silently becomes "must not miss a single seizure". Rather than
    average that away or fail on it, such patients are reported with an exact
    binomial interval (see eval/bootstrap.clopper_pearson_ci) and excluded from
    the gate.
    """
    floor = thresholds.get("min_events_to_gate")
    if floor is None:
        return False
    n_events = flat_metrics.get("worst_patient_n_events")
    return n_events is not None and n_events < floor


def check_gates(flat_metrics: dict[str, float], gates: dict) -> dict:
    results = {}
    for key in flat_metrics:
        if key not in gates or key in NON_GATE_KEYS:
            continue
        thresholds = gates[key]
        if _small_sample_exempt(key, thresholds, flat_metrics):
            results[key] = {
                "value": flat_metrics[key],
                "level": "not_gated_small_sample",
                "n_events": flat_metrics.get("worst_patient_n_events"),
                "min_events_to_gate": thresholds["min_events_to_gate"],
            }
        else:
            results[key] = {"value": flat_metrics[key], "level": _gate_level(flat_metrics[key], thresholds)}
    return results


def _delay_under_alternate_convention(delays_by_patient: dict[str, list[float]], shift_s: float) -> dict:
    """Delay stats as they would read if alarms were credited `shift_s` earlier.

    Exact, not an approximation: the convention is a constant shift applied to
    every alarm timestamp, and the per-event delays are already recorded, so the
    alternate figure is recoverable without re-running anything. Delays stay
    clipped at zero, matching `metrics_event.compute_event_metrics`.
    """
    shifted = [max(0.0, d - shift_s) for delays in delays_by_patient.values() for d in delays]
    if not shifted:
        return {"mean_s": float("nan"), "median_s": float("nan"), "p95_s": float("nan")}
    arr = np.asarray(shifted)
    return {
        "mean_s": float(arr.mean()),
        "median_s": float(np.median(arr)),
        "p95_s": float(np.percentile(arr, 95)),
    }


def build_report(
    per_patient: dict[str, EventMetrics],
    budget: DelayBudget | None = None,
    min_events_to_gate: int | None = None,
) -> dict:
    delays_by_patient = {pid: m.delays_s for pid, m in per_patient.items()}
    macro = macro_mean(per_patient)
    micro = micro_pooled(per_patient)
    delays = delay_stats(per_patient)
    worst = worst_patient(per_patient, min_events=min_events_to_gate)

    sens_ci = clopper_pearson_ci(micro["n_matched"], micro["n_events"])
    far_ci = poisson_rate_ci(micro["n_false_alarms"], micro["exposure_hours"])
    delay_ci = cluster_bootstrap_ci(delays_by_patient, statistic=np.mean)

    if budget is not None:
        # No delay number may be read without the floor it sits on top of.
        delays = {
            **delays,
            "floor_s": budget.floor_s,
            "model_reaction_mean_s": model_reaction_s(delays["mean_s"], budget),
            "model_reaction_median_s": model_reaction_s(delays["median_s"], budget),
            "budget": budget.to_dict(),
            "window_start_convention": _delay_under_alternate_convention(
                delays_by_patient, budget.window_s
            ),
        }

    # Patients excluded from the worst-patient gate are reported with an exact
    # binomial interval rather than being dropped silently: k/n on a 3-seizure
    # case is a point estimate with an interval so wide it carries no
    # information, and saying so is more honest than either gating on it or
    # hiding it.
    small_sample = {
        pid: {
            "n_events": per_patient[pid].n_events,
            "n_matched": per_patient[pid].n_matched,
            "sensitivity": per_patient[pid].sensitivity,
            "sensitivity_ci_95": clopper_pearson_ci(
                per_patient[pid].n_matched, per_patient[pid].n_events
            ),
        }
        for pid in worst.get("patients_below_event_floor", [])
    }

    return {
        "per_patient": {pid: asdict(m) for pid, m in per_patient.items()},
        "small_sample_patients": small_sample,
        "macro": macro,
        "micro": micro,
        "delay": delays,
        "worst_patient": worst,
        "ci_95": {
            "sensitivity_micro": sens_ci,
            "far_per_hour_micro": far_ci,
            "delay_mean_s": delay_ci,
        },
    }


def flatten_for_gates(report: dict) -> dict[str, float]:
    flat = {
        "personalized_event_sensitivity": report["macro"]["sensitivity_macro"],
        "far_per_hour": report["macro"]["far_per_hour_macro"],
        "detection_delay_mean_s": report["delay"]["mean_s"],
        "detection_delay_median_s": report["delay"]["median_s"],
        # Gate on the worst patient the gate is ALLOWED to land on. With no
        # `min_events_to_gate` in the gates file these are the cohort worst,
        # byte-identical to the previous behaviour; with one, they are the
        # worst among patients that have enough seizures for a sensitivity
        # threshold to mean anything (see metrics_event.worst_patient).
        "worst_patient_sensitivity": report["worst_patient"].get(
            "sensitivity_gated", report["worst_patient"]["sensitivity"]
        ),
        "worst_patient_far_per_hour": report["worst_patient"]["far_per_hour"],
        "continuous_test_exposure_hours": report["micro"]["exposure_hours"],
        # Not a gate on the model -- a gate on the measurement setup, so that a
        # delay number can never be judged without the floor it sits on.
        "worst_patient_n_events": report["worst_patient"].get(
            "sensitivity_gated_patient_n_events",
            report["worst_patient"].get("sensitivity_patient_n_events", 0),
        ),
        # The raw cohort worst, always reported, never gated on -- so a
        # small-sample patient can never disappear from the report just
        # because the gate is not allowed to score it.
        "worst_patient_sensitivity_all": report["worst_patient"]["sensitivity"],
    }
    if "floor_s" in report["delay"]:
        flat["detection_delay_floor_s"] = report["delay"]["floor_s"]
    return flat


def save_report(report: dict, path: str) -> None:
    Path(path).write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")

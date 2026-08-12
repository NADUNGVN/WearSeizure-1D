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
from wearseizure.eval.metrics_event import (
    EventMetrics,
    delay_stats,
    macro_mean,
    micro_pooled,
    worst_patient,
)

GATE_LEVELS = ("stretch", "target", "minimum")


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


def check_gates(flat_metrics: dict[str, float], gates: dict) -> dict:
    return {
        key: {"value": flat_metrics[key], "level": _gate_level(flat_metrics[key], gates[key])}
        for key in flat_metrics
        if key in gates
    }


def build_report(per_patient: dict[str, EventMetrics]) -> dict:
    delays_by_patient = {pid: m.delays_s for pid, m in per_patient.items()}
    macro = macro_mean(per_patient)
    micro = micro_pooled(per_patient)
    delays = delay_stats(per_patient)
    worst = worst_patient(per_patient)

    sens_ci = clopper_pearson_ci(micro["n_matched"], micro["n_events"])
    far_ci = poisson_rate_ci(micro["n_false_alarms"], micro["exposure_hours"])
    delay_ci = cluster_bootstrap_ci(delays_by_patient, statistic=np.mean)

    return {
        "per_patient": {pid: asdict(m) for pid, m in per_patient.items()},
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
    return {
        "personalized_event_sensitivity": report["macro"]["sensitivity_macro"],
        "far_per_hour": report["macro"]["far_per_hour_macro"],
        "detection_delay_mean_s": report["delay"]["mean_s"],
        "detection_delay_median_s": report["delay"]["median_s"],
        "worst_patient_sensitivity": report["worst_patient"]["sensitivity"],
        "worst_patient_far_per_hour": report["worst_patient"]["far_per_hour"],
        "continuous_test_exposure_hours": report["micro"]["exposure_hours"],
    }


def save_report(report: dict, path: str) -> None:
    Path(path).write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")

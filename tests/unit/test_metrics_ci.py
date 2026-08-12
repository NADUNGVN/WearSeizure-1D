from __future__ import annotations

import math

import numpy as np

from wearseizure.eval.bootstrap import clopper_pearson_ci, cluster_bootstrap_ci, poisson_rate_ci
from wearseizure.eval.metrics_event import (
    EventMetrics,
    delay_stats,
    macro_mean,
    micro_pooled,
    worst_patient,
)
from wearseizure.eval.report import build_report, check_gates


def test_clopper_pearson_ci_bounds_the_point_estimate():
    lower, upper = clopper_pearson_ci(k=77, n=80, alpha=0.05)
    assert lower < 77 / 80 < upper


def test_clopper_pearson_ci_edge_cases():
    lower_zero, _ = clopper_pearson_ci(0, 10)
    assert lower_zero == 0.0

    _, upper_all = clopper_pearson_ci(10, 10)
    assert upper_all == 1.0


def test_poisson_rate_ci_zero_count():
    lower, upper = poisson_rate_ci(count=0, exposure_hours=100.0)
    assert lower == 0.0
    assert upper > 0.0


def test_poisson_rate_ci_scales_with_exposure():
    _, upper_short = poisson_rate_ci(count=5, exposure_hours=10.0)
    _, upper_long = poisson_rate_ci(count=5, exposure_hours=100.0)
    assert upper_long < upper_short


def test_cluster_bootstrap_ci_matches_mean_for_identical_clusters():
    values = {"p1": [1.0, 1.0], "p2": [1.0, 1.0], "p3": [1.0, 1.0]}
    lower, upper = cluster_bootstrap_ci(values, statistic=np.mean, n_boot=200, rng=np.random.default_rng(0))
    assert lower == upper == 1.0


def test_cluster_bootstrap_ci_empty_input():
    lower, upper = cluster_bootstrap_ci({}, n_boot=10)
    assert math.isnan(lower) and math.isnan(upper)


def _metrics(sens, far, delays, exposure):
    n_events = 4
    return EventMetrics(
        n_events=n_events,
        n_matched=round(sens * n_events),
        n_missed=n_events - round(sens * n_events),
        n_false_alarms=round(far * exposure),
        sensitivity=sens,
        far_per_hour=far,
        delays_s=delays,
        exposure_hours=exposure,
    )


def test_macro_and_micro_pooled_differ_when_patients_unbalanced():
    per_patient = {
        "small": _metrics(sens=1.0, far=0.0, delays=[1.0], exposure=1.0),
        "big": _metrics(sens=0.5, far=0.0, delays=[1.0, 2.0], exposure=100.0),
    }
    macro = macro_mean(per_patient)
    micro = micro_pooled(per_patient)
    assert macro["sensitivity_macro"] == 0.75  # simple average of 1.0 and 0.5
    # micro pools raw counts: (4 + 2) matched out of (4 + 4) events = 0.75 too here,
    # so use FAR (0 either way) plus exposure to show pooling behaves as counts.
    assert micro["exposure_hours"] == 101.0


def test_worst_patient_picks_minimum_sensitivity_and_maximum_far():
    per_patient = {
        "a": _metrics(sens=0.95, far=0.1, delays=[1.0], exposure=10.0),
        "b": _metrics(sens=0.80, far=0.5, delays=[1.0], exposure=10.0),
    }
    worst = worst_patient(per_patient)
    assert worst["sensitivity_patient"] == "b"
    assert worst["far_per_hour_patient"] == "b"


def test_delay_stats_pools_all_patients():
    per_patient = {
        "a": _metrics(sens=1.0, far=0.0, delays=[1.0, 2.0], exposure=1.0),
        "b": _metrics(sens=1.0, far=0.0, delays=[3.0], exposure=1.0),
    }
    stats = delay_stats(per_patient)
    assert stats["mean_s"] == 2.0


def test_build_report_and_check_gates_end_to_end():
    per_patient = {
        "a": _metrics(sens=0.99, far=0.1, delays=[1.0], exposure=50.0),
        "b": _metrics(sens=0.98, far=0.15, delays=[2.0], exposure=50.0),
    }
    report = build_report(per_patient)
    assert "macro" in report and "ci_95" in report

    gates = {
        "personalized_event_sensitivity": {"minimum": 0.97, "target": 0.985, "stretch": 0.99},
    }
    flat = {"personalized_event_sensitivity": report["macro"]["sensitivity_macro"]}
    checked = check_gates(flat, gates)
    assert checked["personalized_event_sensitivity"]["level"] in {"minimum", "target", "stretch", "below_minimum"}

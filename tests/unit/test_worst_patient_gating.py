"""The worst-patient sensitivity gate under the small-sample rule.

`worst_patient_sensitivity` has never been cleared in 26 real-data runs, and
`docs/RESEARCH_REALITY_CHECK.md` section 4 shows why: the gate always lands on
chb17, which has 3 seizures in total, so its sensitivity can only be one of
{0, 1/3, 2/3, 1}. A >=0.85 threshold on such a patient is a demand for a
perfect 3/3, not a sensitivity threshold -- and the reported number says
nothing at all about the other twelve patients.

These tests pin the fix: with `min_events_to_gate` the gate is scored on the
worst patient that has enough seizures for the threshold to be expressible,
while the small-sample patients are still reported, with exact binomial
intervals, rather than hidden.
"""
from __future__ import annotations

import pytest

from wearseizure.eval.metrics_event import EventMetrics, worst_patient
from wearseizure.eval.report import build_report, check_gates, flatten_for_gates


def _metrics(n_events: int, n_matched: int, far_per_hour: float = 0.1) -> EventMetrics:
    return EventMetrics(
        n_events=n_events,
        n_matched=n_matched,
        n_missed=n_events - n_matched,
        n_false_alarms=round(far_per_hour * 10),
        sensitivity=n_matched / n_events,
        far_per_hour=far_per_hour,
        delays_s=[12.0] * n_matched,
        exposure_hours=10.0,
    )


# A cohort shaped like the real one: one 3-seizure patient dragging the
# worst-patient slot down, and several patients with enough seizures to score.
COHORT = {
    "chb01": _metrics(8, 7),      # 0.875
    "chb15": _metrics(20, 16),    # 0.800  <- the worst patient that IS gateable
    "chb17": _metrics(3, 1),      # 0.333  <- 3 seizures, not gateable
    "chb02": _metrics(3, 2),      # 0.667  <- 3 seizures, not gateable
}


def test_without_a_floor_the_gate_lands_on_the_three_seizure_patient():
    worst = worst_patient(COHORT)
    assert worst["sensitivity_patient"] == "chb17"
    assert worst["sensitivity"] == pytest.approx(1 / 3)
    assert worst["patients_below_event_floor"] == []


def test_with_a_floor_the_gate_lands_on_the_worst_gateable_patient():
    worst = worst_patient(COHORT, min_events=5)
    # The raw cohort worst is still reported -- it is never hidden.
    assert worst["sensitivity_patient"] == "chb17"
    assert worst["sensitivity"] == pytest.approx(1 / 3)
    # ...but the gate scores chb15, the worst patient with >=5 seizures.
    assert worst["sensitivity_gated_patient"] == "chb15"
    assert worst["sensitivity_gated"] == pytest.approx(0.80)
    assert worst["sensitivity_gated_patient_n_events"] == 20
    assert worst["patients_below_event_floor"] == ["chb02", "chb17"]


def test_the_gate_scores_the_gateable_worst_not_the_cohort_worst():
    report = build_report(COHORT, min_events_to_gate=5)
    flat = flatten_for_gates(report)
    assert flat["worst_patient_sensitivity"] == pytest.approx(0.80)
    assert flat["worst_patient_sensitivity_all"] == pytest.approx(1 / 3)
    assert flat["worst_patient_n_events"] == 20

    gates = {"worst_patient_sensitivity": {"minimum": 0.85, "min_events_to_gate": 5}}
    result = check_gates(flat, gates)["worst_patient_sensitivity"]
    # 0.80 genuinely fails a 0.85 minimum -- the rule removes an unreachable
    # demand, it does not make the gate free.
    assert result["level"] == "below_minimum"
    assert result["value"] == pytest.approx(0.80)


def test_small_sample_patients_are_reported_with_exact_binomial_intervals():
    report = build_report(COHORT, min_events_to_gate=5)
    small = report["small_sample_patients"]
    assert sorted(small) == ["chb02", "chb17"]
    low, high = small["chb17"]["sensitivity_ci_95"]
    # 1 of 3 -- the interval is nearly the whole unit line, which is the point:
    # a point estimate from 3 trials cannot support a 0.85 threshold.
    assert low < 0.1 and high > 0.9
    assert small["chb17"]["n_events"] == 3


def test_exemption_still_applies_when_no_patient_has_enough_seizures():
    tiny = {"chb17": _metrics(3, 2), "chb02": _metrics(3, 1)}
    report = build_report(tiny, min_events_to_gate=5)
    flat = flatten_for_gates(report)
    gates = {"worst_patient_sensitivity": {"minimum": 0.85, "min_events_to_gate": 5}}
    result = check_gates(flat, gates)["worst_patient_sensitivity"]
    assert result["level"] == "not_gated_small_sample"


def test_report_without_a_floor_is_unchanged():
    # Default path (v1 gates declare no floor) must behave exactly as before.
    report = build_report(COHORT)
    flat = flatten_for_gates(report)
    assert flat["worst_patient_sensitivity"] == pytest.approx(1 / 3)
    assert flat["worst_patient_n_events"] == 3
    assert report["small_sample_patients"] == {}

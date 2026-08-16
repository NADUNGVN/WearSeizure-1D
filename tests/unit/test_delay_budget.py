"""The detection-delay floor imposed by the measurement configuration.

These tests pin the arithmetic behind docs/RESEARCH_REALITY_CHECK.md section 3:
the `detection_delay_mean_s <= 5.0s` gate in configs/eval/gates.yaml is not
reachable under the shipped defaults, because the postprocessor alone already
costs 13.0s before the model contributes anything.
"""
from __future__ import annotations

import numpy as np
import pytest

from wearseizure.eval.delay_budget import delay_budget, ema_group_delay_samples, model_reaction_s
from wearseizure.eval.metrics_event import EventMetrics
from wearseizure.eval.report import build_report, check_gates, flatten_for_gates


def _metrics(delays: list[float], n_events: int = 4) -> EventMetrics:
    return EventMetrics(
        n_events=n_events,
        n_matched=len(delays),
        n_missed=n_events - len(delays),
        n_false_alarms=1,
        sensitivity=len(delays) / n_events,
        far_per_hour=0.5,
        delays_s=delays,
        exposure_hours=2.0,
    )


def test_shipped_defaults_put_the_floor_above_the_v1_gate():
    # configs/window/w4s_stride1s.yaml + configs/postprocess/hysteresis_runlength.yaml
    budget = delay_budget(window_s=4.0, stride_s=1.0, run_length=3, ema_alpha=0.125)
    assert budget.window_term_s == pytest.approx(4.0)
    assert budget.run_length_term_s == pytest.approx(2.0)
    assert budget.ema_term_s == pytest.approx(7.0)
    assert budget.floor_s == pytest.approx(13.0)
    assert budget.floor_s > 5.0, (
        "the v1 detection_delay_mean_s minimum is 5.0s; if the floor ever drops "
        "below it this test should be updated, not deleted"
    )


def test_aggressive_config_from_run_9_reaches_a_five_second_floor():
    # EXPERIMENT_LOG_G1a.md row #9: ema_alpha=0.5, run_length=1
    budget = delay_budget(window_s=4.0, stride_s=1.0, run_length=1, ema_alpha=0.5)
    assert budget.floor_s == pytest.approx(5.0)


def test_proposed_branch_a_config_reaches_the_stretch_floor():
    # w2s window + 0.5s stride + no run-length + light smoothing
    budget = delay_budget(window_s=2.0, stride_s=0.5, run_length=1, ema_alpha=0.5)
    assert budget.floor_s == pytest.approx(2.5)


def test_window_start_convention_credits_back_the_window_term():
    causal = delay_budget(window_s=4.0, stride_s=1.0, run_length=3, ema_alpha=0.125)
    credited = delay_budget(
        window_s=4.0, stride_s=1.0, run_length=3, ema_alpha=0.125, alarm_timestamp="window_start"
    )
    assert credited.window_term_s == pytest.approx(0.0)
    assert credited.floor_s == pytest.approx(causal.floor_s - 4.0)


def test_no_smoothing_contributes_no_ema_term():
    assert ema_group_delay_samples(0.0) == 0.0
    assert ema_group_delay_samples(1.0) == 0.0
    budget = delay_budget(window_s=4.0, stride_s=1.0, run_length=1, ema_alpha=0.0)
    assert budget.floor_s == pytest.approx(4.0)


def test_rejects_unknown_alarm_timestamp_and_bad_geometry():
    with pytest.raises(ValueError):
        delay_budget(window_s=4.0, stride_s=1.0, alarm_timestamp="whenever")
    with pytest.raises(ValueError):
        delay_budget(window_s=0.0, stride_s=1.0)
    with pytest.raises(ValueError):
        delay_budget(window_s=4.0, stride_s=1.0, run_length=0)


def test_model_reaction_is_the_measured_delay_minus_the_floor_clamped_at_zero():
    budget = delay_budget(window_s=4.0, stride_s=1.0, run_length=3, ema_alpha=0.125)
    assert model_reaction_s(19.42, budget) == pytest.approx(6.42)
    assert model_reaction_s(2.0, budget) == 0.0


def test_report_carries_the_floor_and_the_alternate_convention():
    budget = delay_budget(window_s=4.0, stride_s=1.0, run_length=3, ema_alpha=0.125)
    per_patient = {"chb01": _metrics([10.0, 20.0]), "chb02": _metrics([30.0])}

    report = build_report(per_patient, budget=budget)
    assert report["delay"]["floor_s"] == pytest.approx(13.0)
    assert report["delay"]["mean_s"] == pytest.approx(20.0)
    assert report["delay"]["model_reaction_mean_s"] == pytest.approx(7.0)
    # window_start credits 4s back per matched event: (6 + 16 + 26)/3
    assert report["delay"]["window_start_convention"]["mean_s"] == pytest.approx(16.0)
    assert report["delay"]["budget"]["ema_term_s"] == pytest.approx(7.0)

    assert flatten_for_gates(report)["detection_delay_floor_s"] == pytest.approx(13.0)


def test_report_without_a_budget_is_unchanged():
    per_patient = {"chb01": _metrics([10.0])}
    report = build_report(per_patient)
    assert "floor_s" not in report["delay"]
    assert "detection_delay_floor_s" not in flatten_for_gates(report)


def test_worst_patient_gate_is_skipped_for_low_event_patients():
    # chb17 has 3 seizures in CHB-MIT, so its sensitivity can only be one of
    # {0, 1/3, 2/3, 1}; a >=0.85 gate on it means "must be perfect".
    per_patient = {
        "chb01": _metrics([5.0, 6.0, 7.0, 8.0], n_events=4),
        "chb17": _metrics([5.0, 6.0], n_events=3),
    }
    report = build_report(per_patient)
    flat = flatten_for_gates(report)
    assert flat["worst_patient_sensitivity"] == pytest.approx(2 / 3)
    assert flat["worst_patient_n_events"] == 3

    gates = {"worst_patient_sensitivity": {"minimum": 0.85, "min_events_to_gate": 5}}
    result = check_gates(flat, gates)["worst_patient_sensitivity"]
    assert result["level"] == "not_gated_small_sample"
    assert result["n_events"] == 3

    # Without the exemption the same value fails the same threshold.
    strict = check_gates(flat, {"worst_patient_sensitivity": {"minimum": 0.85}})
    assert strict["worst_patient_sensitivity"]["level"] == "below_minimum"


def test_metadata_blocks_in_a_gates_file_are_never_scored():
    flat = {"zero_shot_loso": 0.5, "far_per_hour": 0.1}
    gates = {"zero_shot_loso": {"gated": False}, "far_per_hour": {"minimum": 0.3, "direction": "lower_is_better"}}
    result = check_gates(flat, gates)
    assert "zero_shot_loso" not in result
    assert result["far_per_hour"]["level"] == "minimum"


def test_delay_stats_ignore_nan_free_empty_case():
    report = build_report({"chb01": _metrics([], n_events=2)})
    assert np.isnan(report["delay"]["mean_s"])

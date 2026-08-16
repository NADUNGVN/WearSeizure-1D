from __future__ import annotations

import numpy as np
import pytest

from wearseizure.eval.metrics_event import compute_event_metrics
from wearseizure.postprocess.hysteresis import PostprocessParams
from wearseizure.postprocess.pipeline import run_postprocess
from wearseizure.training.threshold_selection import (
    fit_threshold_on_val,
    fit_threshold_on_val_pooled,
    load_frozen_params,
    save_frozen_params,
)


def _make_scenario():
    end_sec = np.arange(1, 21, dtype=float)
    scores = np.random.default_rng(0).uniform(0.0, 1.0, size=20)
    events = [("e0", 8.5, 11.5)]
    return {"e1": end_sec}, {"e1": scores}, {"e1": events}, {"e1": 1.0}


def _ground_truth(end_sec_by_edf, scores_by_edf, events_by_edf, exposure_by_edf, on, off):
    params = PostprocessParams(
        method="hysteresis_runlength", ema_alpha=1.0, threshold_on=on, threshold_off=off,
        run_length=1, event_merge_gap_s=0.0,
    )
    total_events = total_matched = total_false_alarms = 0
    total_exposure = 0.0
    for edf_id, end_sec in end_sec_by_edf.items():
        alarms = run_postprocess(end_sec, scores_by_edf[edf_id], params)
        m = compute_event_metrics(events_by_edf[edf_id], alarms, exposure_by_edf[edf_id])
        total_events += m.n_events
        total_matched += m.n_matched
        total_false_alarms += m.n_false_alarms
        total_exposure += m.exposure_hours
    sens = total_matched / total_events if total_events else 0.0
    far = total_false_alarms / total_exposure if total_exposure else 0.0
    return sens, far


def test_far_cap_prefers_highest_sensitivity_among_combos_under_the_cap():
    scenario = _make_scenario()
    on_grid = [0.3, 0.4, 0.5, 0.6, 0.7]
    off_grid = [0.1, 0.2, 0.3, 0.4]
    far_cap = 5.0

    ground_truth = {
        (on, off): _ground_truth(*scenario, on, off)
        for on in on_grid for off in off_grid if off < on
    }
    passing = {k: v for k, v in ground_truth.items() if v[1] <= far_cap}
    assert passing, "scenario must have at least one combo under the cap"
    _best_key, (best_sens, best_far) = max(passing.items(), key=lambda kv: (kv[1][0], -kv[1][1]))

    frozen = fit_threshold_on_val(
        *scenario, method="hysteresis_runlength", ema_alpha=1.0, run_length=1, event_merge_gap_s=0.0,
        threshold_on_grid=on_grid, threshold_off_grid=off_grid, fold_id="f0", far_cap_per_hour=far_cap,
    )
    assert frozen.val_sensitivity == pytest.approx(best_sens)
    assert frozen.val_far_per_hour == pytest.approx(best_far)


def test_far_cap_falls_back_to_lowest_far_when_no_combo_qualifies():
    scenario = _make_scenario()
    on_grid = [0.3, 0.4, 0.5]
    off_grid = [0.1, 0.2]
    impossible_cap = -1.0

    ground_truth = {
        (on, off): _ground_truth(*scenario, on, off)
        for on in on_grid for off in off_grid if off < on
    }
    best_far = min(v[1] for v in ground_truth.values())

    frozen = fit_threshold_on_val(
        *scenario, method="hysteresis_runlength", ema_alpha=1.0, run_length=1, event_merge_gap_s=0.0,
        threshold_on_grid=on_grid, threshold_off_grid=off_grid, fold_id="f0", far_cap_per_hour=impossible_cap,
    )
    assert frozen.val_far_per_hour == pytest.approx(best_far)


def test_blend_objective_used_when_far_cap_is_none():
    scenario = _make_scenario()
    on_grid = [0.3, 0.5, 0.7]
    off_grid = [0.1, 0.2]
    far_weight = 0.1

    ground_truth = {
        (on, off): _ground_truth(*scenario, on, off)
        for on in on_grid for off in off_grid if off < on
    }
    _best_key, (best_sens, best_far) = max(ground_truth.items(), key=lambda kv: kv[1][0] - far_weight * kv[1][1])

    frozen = fit_threshold_on_val(
        *scenario, method="hysteresis_runlength", ema_alpha=1.0, run_length=1, event_merge_gap_s=0.0,
        threshold_on_grid=on_grid, threshold_off_grid=off_grid, fold_id="f0",
        far_weight=far_weight, far_cap_per_hour=None,
    )
    assert frozen.val_sensitivity == pytest.approx(best_sens)
    assert frozen.val_far_per_hour == pytest.approx(best_far)


def test_min_delay_objective_trades_surplus_far_headroom_for_speed():
    # One EDF, one seizure at 20-30s. Two threshold levels are available:
    #  - on=0.8 only fires late in the event (slow, but very quiet)
    #  - on=0.3 fires as soon as the event starts (fast, one extra false alarm)
    # Both stay under a generous FAR cap, which is the real situation in
    # EXPERIMENT_LOG_G1a.md: FAR has ~5x more headroom than the gate needs.
    end_sec = np.arange(1, 41, dtype=float)
    scores = np.full(40, 0.1)
    scores[19:30] = 0.5   # the event window: clears 0.3 but not 0.8
    scores[26:30] = 0.9   # only late in the event does it clear 0.8
    scores[34:36] = 0.5   # an interictal blip that only the low threshold trips
    scenario = ({"e1": end_sec}, {"e1": scores}, {"e1": [("e0", 20.0, 30.0)]}, {"e1": 1.0})
    grids = dict(threshold_on_grid=[0.3, 0.8], threshold_off_grid=[0.2])
    common = dict(
        method="hysteresis_runlength", ema_alpha=1.0, run_length=1,
        event_merge_gap_s=0.0, fold_id="f0", far_cap_per_hour=10.0, **grids,
    )

    fastest = fit_threshold_on_val(*scenario, objective="min_delay", **common)
    quietest = fit_threshold_on_val(*scenario, objective="max_sensitivity", **common)

    assert fastest.params.threshold_on == 0.3
    assert fastest.val_delay_mean_s < quietest.val_delay_mean_s
    assert fastest.val_far_per_hour >= quietest.val_far_per_hour, (
        "the whole point is that speed is bought with FAR headroom"
    )


def test_min_delay_respects_the_sensitivity_floor():
    end_sec = np.arange(1, 41, dtype=float)
    scores = np.full(40, 0.1)
    scores[19:30] = 0.5
    scores[26:30] = 0.9
    scores[34:36] = 0.5
    scenario = ({"e1": end_sec}, {"e1": scores}, {"e1": [("e0", 20.0, 30.0)]}, {"e1": 1.0})
    common = dict(
        method="hysteresis_runlength", ema_alpha=1.0, run_length=1, event_merge_gap_s=0.0,
        threshold_on_grid=[0.3, 0.8], threshold_off_grid=[0.2], fold_id="f0",
        far_cap_per_hour=10.0, objective="min_delay",
    )
    # An unreachable floor must not silently win: it is dropped before the FAR
    # cap is, so a candidate under the cap is still returned.
    frozen = fit_threshold_on_val(*scenario, sensitivity_floor=2.0, **common)
    assert frozen.val_far_per_hour <= 10.0


def test_rejects_unknown_objective():
    scenario = _make_scenario()
    with pytest.raises(ValueError):
        fit_threshold_on_val(
            *scenario, method="hysteresis_runlength", ema_alpha=1.0, run_length=1,
            event_merge_gap_s=0.0, threshold_on_grid=[0.3, 0.5], threshold_off_grid=[0.1],
            fold_id="f0", objective="be_perfect",
        )


def test_default_objective_still_records_the_delay_it_paid():
    scenario = _make_scenario()
    frozen = fit_threshold_on_val(
        *scenario, method="hysteresis_runlength", ema_alpha=1.0, run_length=1, event_merge_gap_s=0.0,
        threshold_on_grid=[0.3, 0.5], threshold_off_grid=[0.1, 0.2], fold_id="f0", far_cap_per_hour=5.0,
    )
    assert frozen.val_delay_mean_s == frozen.val_delay_mean_s, "delay must be recorded, not NaN"


def test_rejects_unsupported_method():
    with pytest.raises(NotImplementedError):
        fit_threshold_on_val(
            {}, {}, {}, {}, method="raw_threshold", ema_alpha=0.1, run_length=1, event_merge_gap_s=0.0,
            threshold_on_grid=[0.5], threshold_off_grid=[0.3], fold_id="f0",
        )


def test_pooled_matches_single_fold_when_given_only_one_fold():
    scenario = _make_scenario()
    on_grid, off_grid = [0.3, 0.5, 0.7], [0.1, 0.2]

    single = fit_threshold_on_val(
        *scenario, method="hysteresis_runlength", ema_alpha=1.0, run_length=1, event_merge_gap_s=0.0,
        threshold_on_grid=on_grid, threshold_off_grid=off_grid, fold_id="f0", far_cap_per_hour=5.0,
    )
    pooled = fit_threshold_on_val_pooled(
        val_folds=[scenario], method="hysteresis_runlength", ema_alpha=1.0, run_length=1, event_merge_gap_s=0.0,
        threshold_on_grid=on_grid, threshold_off_grid=off_grid, group_id="patientX", far_cap_per_hour=5.0,
    )
    assert pooled.val_sensitivity == pytest.approx(single.val_sensitivity)
    assert pooled.val_far_per_hour == pytest.approx(single.val_far_per_hour)
    assert pooled.fold_id == "patientX"


def test_pooling_rescues_a_fold_whose_own_validation_never_fires():
    # Fold A: score never rises above the lowest grid threshold (the exact
    # chb17-style failure mode found in the real-data run: val_sensitivity=0
    # at every available threshold because there's only one val event and
    # the model's confidence for it never clears 0.2).
    end_sec = np.arange(1, 11, dtype=float)
    fold_a = (
        {"a1": end_sec}, {"a1": np.full(10, 0.1)}, {"a1": [("ea", 4.5, 6.5)]}, {"a1": 1.0},
    )
    # Fold B (different EDF, same patient, model confidently separates it).
    scores_b = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1])
    fold_b = (
        {"b1": end_sec}, {"b1": scores_b}, {"b1": [("eb", 4.5, 6.5)]}, {"b1": 1.0},
    )
    on_grid, off_grid = [0.2, 0.5, 0.8], [0.1, 0.3]

    alone_a = fit_threshold_on_val(
        *fold_a, method="hysteresis_runlength", ema_alpha=1.0, run_length=1, event_merge_gap_s=0.0,
        threshold_on_grid=on_grid, threshold_off_grid=off_grid, fold_id="a", far_cap_per_hour=5.0,
    )
    assert alone_a.val_sensitivity == 0.0, "fold A alone should never detect its one event at any grid threshold"

    pooled = fit_threshold_on_val_pooled(
        val_folds=[fold_a, fold_b], method="hysteresis_runlength", ema_alpha=1.0, run_length=1,
        event_merge_gap_s=0.0, threshold_on_grid=on_grid, threshold_off_grid=off_grid,
        group_id="patientX", far_cap_per_hour=5.0,
    )
    assert pooled.val_sensitivity == pytest.approx(0.5)  # 1 of 2 pooled events detected via fold B
    assert pooled.val_far_per_hour == pytest.approx(0.0)


def test_pooled_rejects_unsupported_method():
    with pytest.raises(NotImplementedError):
        fit_threshold_on_val_pooled(
            [], method="raw_threshold", ema_alpha=0.1, run_length=1, event_merge_gap_s=0.0,
            threshold_on_grid=[0.5], threshold_off_grid=[0.3], group_id="p0",
        )


def test_pooled_rejects_empty_fold_list():
    with pytest.raises(ValueError):
        fit_threshold_on_val_pooled(
            [], method="hysteresis_runlength", ema_alpha=0.1, run_length=1, event_merge_gap_s=0.0,
            threshold_on_grid=[0.5], threshold_off_grid=[0.3], group_id="p0",
        )


def test_save_and_load_frozen_params_roundtrip(tmp_path):
    scenario = _make_scenario()
    frozen = fit_threshold_on_val(
        *scenario, method="hysteresis_runlength", ema_alpha=1.0, run_length=1, event_merge_gap_s=0.0,
        threshold_on_grid=[0.3, 0.5], threshold_off_grid=[0.1, 0.2], fold_id="f0", far_cap_per_hour=1.0,
    )
    path = tmp_path / "frozen.json"
    save_frozen_params(frozen, str(path))
    loaded = load_frozen_params(str(path))
    assert loaded == frozen

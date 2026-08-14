from __future__ import annotations

import numpy as np
import pytest

from wearseizure.eval.metrics_event import compute_event_metrics
from wearseizure.postprocess.hysteresis import PostprocessParams
from wearseizure.postprocess.pipeline import run_postprocess
from wearseizure.training.threshold_selection import (
    fit_threshold_on_val,
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


def test_rejects_unsupported_method():
    with pytest.raises(NotImplementedError):
        fit_threshold_on_val(
            {}, {}, {}, {}, method="raw_threshold", ema_alpha=0.1, run_length=1, event_merge_gap_s=0.0,
            threshold_on_grid=[0.5], threshold_off_grid=[0.3], fold_id="f0",
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

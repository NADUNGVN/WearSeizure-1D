"""Paired cluster bootstrap between two configurations.

The project's central claim is comparative: detection equivalent-or-better than
the reproduced baselines at roughly 4x lower compute. Its three best
configurations sit 1.4pp apart on 77 seizures -- about one seizure -- so that
claim cannot be made from point estimates. These tests pin that the machinery
says "indistinguishable" when there is no real difference and only claims a
winner when there is one.
"""
from __future__ import annotations

# scripts/ is not an importable package; load the script by path.
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from wearseizure.eval.bootstrap import paired_cluster_bootstrap

_spec = importlib.util.spec_from_file_location(
    "paired_bootstrap_script",
    Path(__file__).resolve().parents[2] / "scripts" / "paired_bootstrap.py",
)
paired_bootstrap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(paired_bootstrap)


PATIENTS = [f"chb{i:02d}" for i in range(1, 14)]


def _delta(a: dict, b: dict):
    return lambda ids: float(np.mean([a[i] for i in ids]) - np.mean([b[i] for i in ids]))


def test_a_consistent_edge_produces_an_interval_that_excludes_zero():
    a = {p: 0.90 for p in PATIENTS}
    b = {p: 0.80 for p in PATIENTS}
    result = paired_cluster_bootstrap(PATIENTS, _delta(a, b), n_boot=1000)
    assert result["delta"] == pytest.approx(0.10)
    assert result["ci_low"] > 0
    assert result["n_clusters"] == 13


def test_no_systematic_difference_produces_an_interval_that_contains_zero():
    # Deliberately symmetric rather than random: six patients where A wins by
    # 0.05, six where B wins by 0.05, one tie. Per-patient differences are as
    # large as anything seen in the real logs, but they cancel, so the cohort
    # difference is zero and the interval must say so. Drawing both sides from
    # a uniform instead would sometimes produce a genuine gap by chance and
    # make this test flaky for the wrong reason.
    a, b = {}, {}
    for i, patient in enumerate(PATIENTS):
        edge = 0.05 if i % 2 == 0 and i < 12 else (-0.05 if i < 12 else 0.0)
        a[patient] = 0.85 + edge
        b[patient] = 0.85
    result = paired_cluster_bootstrap(PATIENTS, _delta(a, b), n_boot=4000)
    assert result["delta"] == pytest.approx(0.0, abs=1e-12)
    assert result["ci_low"] < 0 < result["ci_high"]


def test_an_edge_carried_by_one_patient_is_not_called_significant():
    # This is exactly why the cluster is the patient. One patient out of 13
    # doing much better cannot, on its own, support a cohort-level claim --
    # resampling patients means that patient is absent from ~28% of replicates.
    a = {p: 0.80 for p in PATIENTS}
    b = {p: 0.80 for p in PATIENTS}
    a["chb01"] = 1.00
    result = paired_cluster_bootstrap(PATIENTS, _delta(a, b), n_boot=4000)
    assert result["delta"] > 0
    assert result["ci_low"] <= 0, "a single patient must not be enough to exclude zero"


def test_bootstrap_is_deterministic_for_a_given_rng_seed():
    a = {p: 0.9 for p in PATIENTS}
    b = {p: 0.8 for p in PATIENTS}
    kwargs = {"n_boot": 500}
    first = paired_cluster_bootstrap(PATIENTS, _delta(a, b), rng=np.random.default_rng(0), **kwargs)
    second = paired_cluster_bootstrap(PATIENTS, _delta(a, b), rng=np.random.default_rng(0), **kwargs)
    assert first == second


def test_empty_cluster_list_returns_nan_rather_than_raising():
    result = paired_cluster_bootstrap([], lambda ids: 0.0, n_boot=10)
    assert np.isnan(result["delta"])
    assert result["n_clusters"] == 0


# --------------------------------------------------------------------------
# The script's own loading / statistic layer
# --------------------------------------------------------------------------


def _write_fold(run_dir: Path, fold_id: str, n_events: int, n_matched: int,
                n_false_alarms: int = 1, exposure_hours: float = 3.0) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"{fold_id}.metrics.json").write_text(
        json.dumps(
            {
                "fold_id": fold_id,
                "test_event_metrics": {
                    "n_events": n_events,
                    "n_matched": n_matched,
                    "n_false_alarms": n_false_alarms,
                    "exposure_hours": exposure_hours,
                    "delays_s": [10.0] * n_matched,
                },
            }
        ),
        encoding="utf-8",
    )


def test_seed_directories_are_averaged_into_one_configuration(tmp_path: Path):
    window = tmp_path / "w4s"
    # Same fold, two seeds: 2/4 and 4/4 matched -> the configuration is 3/4.
    _write_fold(window / "seed0", "chb01__chb01_03", n_events=4, n_matched=2)
    _write_fold(window / "seed1", "chb01__chb01_03", n_events=4, n_matched=4)

    per_patient = paired_bootstrap._load_per_patient(window, "patient_specific_loso_edf")
    assert per_patient["chb01"]["n_seeds"] == 2
    assert per_patient["chb01"]["n_matched"] == pytest.approx(3.0)
    assert paired_bootstrap.statistic(per_patient, ["chb01"], "sensitivity_macro") == pytest.approx(0.75)


def test_a_directory_without_seed_subdirs_is_read_as_a_single_seed(tmp_path: Path):
    run_dir = tmp_path / "seed0"
    _write_fold(run_dir, "chb01__chb01_03", n_events=4, n_matched=3)
    per_patient = paired_bootstrap._load_per_patient(run_dir, "patient_specific_loso_edf")
    assert per_patient["chb01"]["n_seeds"] == 1


def test_macro_and_micro_sensitivity_differ_when_patients_are_unbalanced(tmp_path: Path):
    run_dir = tmp_path / "seed0"
    _write_fold(run_dir, "chb01__chb01_03", n_events=2, n_matched=2)    # 1.00
    _write_fold(run_dir, "chb15__chb15_10", n_events=20, n_matched=10)  # 0.50
    per_patient = paired_bootstrap._load_per_patient(run_dir, "patient_specific_loso_edf")
    ids = ["chb01", "chb15"]
    # Macro weights patients equally; micro pools seizures, so the 20-seizure
    # patient dominates. Reporting only one of the two hides that difference.
    assert paired_bootstrap.statistic(per_patient, ids, "sensitivity_macro") == pytest.approx(0.75)
    assert paired_bootstrap.statistic(per_patient, ids, "sensitivity_micro") == pytest.approx(12 / 22)


# --------------------------------------------------------------------------
# Verdict direction. Getting this wrong turns a win into a loss in the one
# line a reader actually looks at.
# --------------------------------------------------------------------------


def test_lower_is_better_metrics_credit_a_negative_delta_to_a():
    # The real Phase 1 numbers: k5only FAR 0.2577/h vs default 0.3658/h, an
    # interval that excludes zero. delta is A-B and therefore negative, which
    # an unqualified `delta > 0` test reported as "B better".
    assert paired_bootstrap.favours_a("far_per_hour_micro", -0.1081, False) == "A"
    assert paired_bootstrap.favours_a("far_per_hour_micro", +0.1081, False) == "B"
    assert paired_bootstrap.favours_a("delay_mean_s", +0.5163, False) == "B"
    assert paired_bootstrap.favours_a("delay_mean_s", -0.5163, False) == "A"


def test_higher_is_better_metrics_credit_a_positive_delta_to_a():
    assert paired_bootstrap.favours_a("sensitivity_macro", +0.0115, False) == "A"
    assert paired_bootstrap.favours_a("sensitivity_micro", -0.0043, False) == "B"


def test_an_interval_containing_zero_favours_neither_whatever_the_sign():
    for metric in paired_bootstrap.METRICS:
        assert paired_bootstrap.favours_a(metric, +1.0, True) == "neither"
        assert paired_bootstrap.favours_a(metric, -1.0, True) == "neither"


def test_every_reported_metric_declares_a_direction():
    assert set(paired_bootstrap.METRICS) == set(paired_bootstrap.METRIC_DIRECTION)


def test_only_seed_digit_directories_count_as_seeds(tmp_path: Path):
    # The Phase 2 scripts park an un-pre-trained run in `seed0_noL1` before
    # writing the pre-trained one into `seed0`. A plain `seed*` glob swept that
    # backup up as a fourth seed and averaged a no-L1 run into a 3-seed L1
    # result -- which moved baseline_frontiers2d from 0.9726 to 0.9591 and
    # invalidated both architecture comparisons without erroring.
    window = tmp_path / "w4s"
    for name in ("seed0", "seed1", "seed2"):
        _write_fold(window / name, "chb01__chb01_03", n_events=4, n_matched=4)
    _write_fold(window / "seed0_noL1", "chb01__chb01_03", n_events=4, n_matched=0)

    per_patient = paired_bootstrap._load_per_patient(window, "patient_specific_loso_edf")
    assert per_patient["chb01"]["n_seeds"] == 3
    # 4/4 on every real seed. Including the backup would have given 3/4.
    assert paired_bootstrap.statistic(per_patient, ["chb01"], "sensitivity_macro") == pytest.approx(1.0)


def test_double_digit_seeds_are_not_dropped(tmp_path: Path):
    window = tmp_path / "w4s"
    for name in ("seed0", "seed10"):
        _write_fold(window / name, "chb01__chb01_03", n_events=4, n_matched=2)
    assert paired_bootstrap._load_per_patient(window, "patient_specific_loso_edf")["chb01"]["n_seeds"] == 2


# --------------------------------------------------------------------------
# Worst-patient axes. This is where the architectures differ most, so it is
# the axis most in need of an interval rather than a point estimate.
# --------------------------------------------------------------------------


def test_worst_patient_far_takes_the_maximum(tmp_path: Path):
    run_dir = tmp_path / "seed0"
    _write_fold(run_dir, "chb01__chb01_03", 4, 4, n_false_alarms=1, exposure_hours=10.0)   # 0.1/h
    _write_fold(run_dir, "chb11__chb11_82", 6, 5, n_false_alarms=9, exposure_hours=3.0)    # 3.0/h
    per_patient = paired_bootstrap._load_per_patient(run_dir, "patient_specific_loso_edf")
    v = paired_bootstrap.statistic(per_patient, ["chb01", "chb11"], "worst_patient_far_per_hour")
    assert v == pytest.approx(3.0)


def test_worst_patient_sensitivity_respects_the_seizure_floor(tmp_path: Path):
    run_dir = tmp_path / "seed0"
    _write_fold(run_dir, "chb15__chb15_10", 20, 16)   # 0.80, gateable
    _write_fold(run_dir, "chb17__chb17_63", 3, 1)     # 0.333, below the floor
    per_patient = paired_bootstrap._load_per_patient(run_dir, "patient_specific_loso_edf")
    ids = ["chb15", "chb17"]
    assert paired_bootstrap.statistic(per_patient, ids, "worst_patient_sensitivity") == pytest.approx(0.80)
    # Drop the floor and the 3-seizure patient takes the slot back.
    assert paired_bootstrap.statistic(
        per_patient, ids, "worst_patient_sensitivity", min_events=1) == pytest.approx(1 / 3)


def test_far_has_no_seizure_floor(tmp_path: Path):
    # A rate over hours is meaningful for every patient, unlike a count over
    # events -- so the floor must not silently apply to FAR as well.
    run_dir = tmp_path / "seed0"
    _write_fold(run_dir, "chb17__chb17_63", 3, 1, n_false_alarms=12, exposure_hours=4.0)  # 3.0/h
    _write_fold(run_dir, "chb15__chb15_10", 20, 16, n_false_alarms=1, exposure_hours=10.0)
    per_patient = paired_bootstrap._load_per_patient(run_dir, "patient_specific_loso_edf")
    v = paired_bootstrap.statistic(per_patient, ["chb17", "chb15"], "worst_patient_far_per_hour")
    assert v == pytest.approx(3.0)


def test_worst_patient_metrics_declare_a_direction():
    assert paired_bootstrap.METRIC_DIRECTION["worst_patient_far_per_hour"] == "lower_is_better"
    assert paired_bootstrap.METRIC_DIRECTION["worst_patient_sensitivity"] == "higher_is_better"
    assert paired_bootstrap.favours_a("worst_patient_far_per_hour", -1.44, False) == "A"


# --------------------------------------------------------------------------
# Paired per-patient comparison. The bootstrap cannot resolve a max over 13
# clusters -- its distribution collapses onto "is the worst patient in this
# replicate?" -- so the tail argument needs a statistic that is not a min/max.
# --------------------------------------------------------------------------


def test_sign_test_detects_a_consistent_far_advantage():
    rows = [{"far_a": 0.1 * i, "far_b": 0.1 * i + 0.5} for i in range(1, 14)]
    st = paired_bootstrap.sign_test(rows, "far_a", "far_b", lower_is_better=True)
    assert st["wins_a"] == 13 and st["wins_b"] == 0
    assert st["p_value"] < 0.001


def test_sign_test_reports_no_effect_when_patients_split_evenly():
    rows = [{"far_a": 1.0, "far_b": 2.0} for _ in range(6)]
    rows += [{"far_a": 2.0, "far_b": 1.0} for _ in range(6)]
    st = paired_bootstrap.sign_test(rows, "far_a", "far_b", lower_is_better=True)
    assert st["wins_a"] == 6 and st["wins_b"] == 6
    assert st["p_value"] == pytest.approx(1.0)


def test_sign_test_drops_ties_rather_than_counting_them():
    # Standard sign-test convention: a patient where both configurations are
    # identical carries no evidence either way, so it must not dilute the count.
    rows = [{"far_a": 1.0, "far_b": 2.0}] * 5 + [{"far_a": 1.0, "far_b": 1.0}] * 4
    st = paired_bootstrap.sign_test(rows, "far_a", "far_b", lower_is_better=True)
    assert (st["wins_a"], st["wins_b"], st["ties"], st["n"]) == (5, 0, 4, 5)


def test_sign_test_honours_direction():
    rows = [{"sens_a": 0.9, "sens_b": 0.8} for _ in range(10)]
    st = paired_bootstrap.sign_test(rows, "sens_a", "sens_b", lower_is_better=False)
    assert st["wins_a"] == 10


def test_a_nan_patient_is_treated_as_a_tie_not_a_win():
    rows = [{"far_a": float("nan"), "far_b": 2.0}, {"far_a": 1.0, "far_b": 2.0}]
    st = paired_bootstrap.sign_test(rows, "far_a", "far_b", lower_is_better=True)
    assert (st["wins_a"], st["ties"]) == (1, 1)

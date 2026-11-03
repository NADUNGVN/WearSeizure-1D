"""The power measurement must refuse to report a difference it cannot see.

Only the arithmetic is tested here -- the I2C side needs the board. What makes
these worth asserting is that the failure mode is silent: a differential
measurement always produces a number, and a number produced from two baselines
that differ by more than the effect is indistinguishable, on paper, from a
real result. The script's job is to notice that and say so.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location(
        "board_power", REPO / "scripts" / "board_power.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["board_power"] = mod
    spec.loader.exec_module(mod)
    return mod


bp = _load()


def rows(values):
    return [{"p": v} for v in values]


def test_a_clear_effect_is_reported_with_its_energy():
    before = rows([5.00, 5.01, 4.99, 5.00])
    after = rows([5.01, 5.00, 5.00, 4.99])
    active = rows([5.40, 5.41, 5.39, 5.40])
    d = bp.differential(before, after, active, "p", wall_s=100.0,
                        inferences=1000, macs=489_472)
    assert d["credible"] is True
    assert abs(d["delta_W"] - 0.40) < 1e-6
    # 0.4 W over 100 s shared by 1000 inferences is 40 mJ each.
    assert abs(d["energy_per_inference_J"] - 0.04) < 1e-9
    assert abs(d["pJ_per_MAC"] - 0.04 / 489_472 * 1e12) < 1.0


def test_an_effect_inside_the_noise_is_refused():
    before = rows([5.00, 5.20, 4.80, 5.00])     # sd ~0.14
    after = rows([5.00, 5.15, 4.85, 5.00])
    active = rows([5.05, 5.06, 5.04, 5.05])     # delta ~0.05, well inside
    d = bp.differential(before, after, active, "p", 100.0, 1000, 489_472)
    assert d["credible"] is False


def test_baseline_drift_counts_as_noise():
    # Both baselines are internally steady, so their spreads are ~0. If drift
    # were not in the floor, a board that simply warmed up by 0.3 W would
    # report 0.3 W of "dynamic power" with an apparently infinite margin.
    before = rows([5.00, 5.00, 5.00, 5.00])
    after = rows([5.30, 5.30, 5.30, 5.30])
    active = rows([5.35, 5.35, 5.35, 5.35])
    d = bp.differential(before, after, active, "p", 100.0, 1000, 489_472)
    assert abs(d["baseline_drift_W"] - 0.30) < 1e-9
    assert abs(d["noise_floor_W"] - 0.30) < 1e-9
    assert d["credible"] is False


def test_idle_is_the_mean_of_both_baselines_not_the_first():
    before = rows([5.00] * 4)
    after = rows([5.40] * 4)
    active = rows([9.00] * 4)
    d = bp.differential(before, after, active, "p", 100.0, None, 489_472)
    assert abs(d["idle_W"] - 5.20) < 1e-9
    assert abs(d["delta_W"] - 3.80) < 1e-9


def test_without_an_inference_count_no_energy_is_invented():
    d = bp.differential(rows([5.0] * 4), rows([5.0] * 4), rows([6.0] * 4),
                        "p", 100.0, None, 489_472)
    assert "energy_per_inference_J" not in d
    assert "pJ_per_MAC" not in d


def test_a_zero_noise_floor_is_not_treated_as_infinite_confidence():
    # Four identical readings give sd 0 and drift 0. A ratio against zero is
    # not evidence; it is a quantiser that did not move.
    d = bp.differential(rows([5.0] * 4), rows([5.0] * 4), rows([5.0] * 4),
                        "p", 100.0, 1000, 489_472)
    assert d["noise_floor_W"] == 0.0
    assert d["credible"] is False
    assert d["delta_over_noise"] is None

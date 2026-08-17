"""The delay floor must come from the params the folds were thresholded with.

`rethreshold.py` takes `postprocess.run_length=... postprocess.ema_alpha=...`
and writes into the same `*.metrics.json` files; `evaluate.py` is then normally
run without repeating those overrides. Reading the floor from `cfg.postprocess`
therefore reports a floor for a configuration that never ran -- observed once
as a 5.0s floor reported as 13.0s, which inflated "model reaction" by 8s.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

_spec = importlib.util.spec_from_file_location(
    "_evaluate_script", Path(__file__).resolve().parents[2] / "scripts" / "evaluate.py"
)
_evaluate = importlib.util.module_from_spec(_spec)
sys.modules["_evaluate_script"] = _evaluate
_spec.loader.exec_module(_evaluate)


def _cfg(run_length=3, ema_alpha=0.125):
    return OmegaConf.create(
        {"window": {"window_s": 4.0, "stride_s": 1.0},
         "postprocess": {"run_length": run_length, "ema_alpha": ema_alpha}}
    )


def _fold(run_length, ema_alpha, alarm_timestamp="window_end"):
    return {"frozen_postprocess": {"params": {
        "run_length": run_length, "ema_alpha": ema_alpha, "alarm_timestamp": alarm_timestamp}}}


def test_frozen_params_win_over_a_stale_config():
    # The exact situation that produced the wrong number: folds re-thresholded
    # at L=1/alpha=0.5, evaluate.py invoked with the L=3/alpha=0.125 defaults.
    folds = [_fold(1, 0.5) for _ in range(3)]
    budget = _evaluate._budget_from_frozen_params(folds, _cfg(run_length=3, ema_alpha=0.125))
    assert budget.floor_s == pytest.approx(5.0)


def test_matching_config_gives_the_same_answer():
    folds = [_fold(3, 0.125) for _ in range(3)]
    budget = _evaluate._budget_from_frozen_params(folds, _cfg(3, 0.125))
    assert budget.floor_s == pytest.approx(13.0)


def test_mixed_postprocess_settings_across_folds_are_refused():
    folds = [_fold(3, 0.125), _fold(1, 0.5)]
    with pytest.raises(RuntimeError, match="different postprocess settings"):
        _evaluate._budget_from_frozen_params(folds, _cfg())


def test_falls_back_to_config_when_metrics_predate_the_field():
    budget = _evaluate._budget_from_frozen_params([{}, {}], _cfg(3, 0.125))
    assert budget.floor_s == pytest.approx(13.0)

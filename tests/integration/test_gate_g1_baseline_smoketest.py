"""Smoke-test both reproduction baselines (memo 7.1 #1-#2) run end-to-end
through the same fold/engine machinery WearSeizure-1D uses.
"""
from __future__ import annotations

import pytest

from wearseizure.data.splits import make_patient_specific_loso_edf
from wearseizure.models.baselines import Compact1DBaseline, FrontiersBaseline2D
from wearseizure.training.engine_baseline import run_fold


@pytest.mark.integration
@pytest.mark.parametrize("model_cls", [Compact1DBaseline, FrontiersBaseline2D])
def test_baseline_runs_end_to_end_on_synthetic(synthetic_cohort, model_cls):
    manifest_df, records = synthetic_cohort
    folds = make_patient_specific_loso_edf(manifest_df, seed=0)
    fold = folds[0]

    model = model_cls(input_len=1024)
    result = run_fold(
        model=model,
        records=records,
        fold=fold,
        window_s=4.0,
        stride_s=1.0,
        postprocess_method="hysteresis_runlength",
        postprocess_ema_alpha=0.125,
        postprocess_run_length=2,
        postprocess_event_merge_gap_s=1.0,
        threshold_on_grid=[0.5, 0.6],
        threshold_off_grid=[0.3, 0.4],
        epochs=1,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=32,
        device="cpu",
        class_balanced_sampling=True,
        early_stopping_patience=1,
    )
    assert 0.0 <= result.test_segment_metrics.prevalence <= 1.0
    assert result.frozen_postprocess.params.method == "hysteresis_runlength"

"""End-to-end smoke test on synthetic data: split -> train a couple epochs on
CPU -> freeze threshold on val -> evaluate continuous test -> build report.
Not a claim about clinical performance (synthetic data has none) -- only
proves the wiring between every G0-G1 module is correct.
"""
from __future__ import annotations

import pytest

from wearseizure.data.splits import make_patient_specific_loso_edf
from wearseizure.eval.report import build_report
from wearseizure.models.wearseizure1d import WearSeizure1D
from wearseizure.training.engine_baseline import run_fold


@pytest.mark.integration
def test_full_pipeline_smoketest_on_synthetic(synthetic_cohort):
    manifest_df, records = synthetic_cohort
    folds = make_patient_specific_loso_edf(manifest_df, seed=0)
    fold = folds[0]

    model = WearSeizure1D(input_len=1024)
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
        epochs=2,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=32,
        device="cpu",
        class_balanced_sampling=True,
        early_stopping_patience=2,
    )

    assert result.test_event_metrics.n_events >= 0
    assert 0.0 <= result.test_segment_metrics.prevalence <= 1.0

    report = build_report({fold.held_out_key: result.test_event_metrics})
    assert "macro" in report
    assert "ci_95" in report

from __future__ import annotations

import numpy as np

from wearseizure.data.manifest import events_for_row
from wearseizure.data.synthetic import generate_synthetic_cohort, generate_synthetic_record


def test_generate_synthetic_record_shape_matches_meta():
    rng = np.random.default_rng(0)
    record = generate_synthetic_record(
        subject_id="synA", edf_id="synA_00", duration_s=10.0, rng=rng, fs_hz=256.0,
        seizure_intervals=[(2.0, 5.0)],
    )
    assert record.signal.shape == (2560,)
    assert record.meta.n_samples == 2560
    assert len(record.meta.seizure_events) == 1
    assert record.meta.seizure_events[0].onset_sec == 2.0


def test_generate_synthetic_record_is_deterministic_given_same_rng_state():
    r1 = generate_synthetic_record(
        subject_id="s", edf_id="e", duration_s=5.0, rng=np.random.default_rng(42),
    )
    r2 = generate_synthetic_record(
        subject_id="s", edf_id="e", duration_s=5.0, rng=np.random.default_rng(42),
    )
    np.testing.assert_array_equal(r1.signal, r2.signal)


def test_generate_synthetic_cohort_is_deterministic_given_seed():
    df1, _ = generate_synthetic_cohort(n_subjects=3, edfs_per_subject=2, seed=7, edf_duration_s=30.0)
    df2, _ = generate_synthetic_cohort(n_subjects=3, edfs_per_subject=2, seed=7, edf_duration_s=30.0)
    assert df1.equals(df2)


def test_generate_synthetic_cohort_has_both_ictal_and_interictal_edfs(synthetic_cohort):
    manifest_df, _records = synthetic_cohort
    has_events = [len(events_for_row(row)) > 0 for _, row in manifest_df.iterrows()]
    assert any(has_events), "expected at least one ictal EDF"
    assert not all(has_events), "expected at least one interictal-only EDF"
    assert manifest_df["subject_id"].nunique() >= 3


def test_generate_synthetic_cohort_edf_ids_are_unique(synthetic_cohort):
    manifest_df, records = synthetic_cohort
    assert manifest_df["edf_id"].is_unique
    assert set(manifest_df["edf_id"]) == set(records.keys())

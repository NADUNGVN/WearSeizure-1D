from __future__ import annotations

import pytest

from wearseizure.data.manifest import events_for_row
from wearseizure.data.splits import make_patient_specific_loso_edf, make_zero_shot_loso_subject


def test_zero_shot_loso_produces_one_fold_per_subject(synthetic_cohort):
    manifest_df, _ = synthetic_cohort
    subjects = set(manifest_df["subject_id"])
    folds = make_zero_shot_loso_subject(manifest_df, seed=0)
    assert {f.held_out_key for f in folds} == subjects
    for fold in folds:
        expected_test = set(manifest_df.loc[manifest_df.subject_id == fold.held_out_key, "edf_id"])
        assert set(fold.test_edf_ids) == expected_test


def test_zero_shot_loso_requires_at_least_three_subjects():
    import pandas as pd

    tiny = pd.DataFrame(
        [
            {"subject_id": "a", "edf_id": "a0", "seizure_events": "[]"},
            {"subject_id": "b", "edf_id": "b0", "seizure_events": "[]"},
        ]
    )
    with pytest.raises(ValueError, match="at least 3 subjects"):
        make_zero_shot_loso_subject(tiny, seed=0)


def test_patient_specific_loso_one_fold_per_ictal_edf(synthetic_cohort):
    manifest_df, _ = synthetic_cohort
    n_ictal = sum(len(events_for_row(row)) > 0 for _, row in manifest_df.iterrows())
    folds = make_patient_specific_loso_edf(manifest_df, seed=0)
    assert len(folds) == n_ictal


def test_patient_specific_loso_test_partition_contains_the_ictal_edf(synthetic_cohort):
    manifest_df, _ = synthetic_cohort
    folds = make_patient_specific_loso_edf(manifest_df, seed=0)
    for fold in folds:
        assert fold.held_out_key in fold.test_edf_ids


def test_splits_are_deterministic_given_same_seed(synthetic_cohort):
    manifest_df, _ = synthetic_cohort
    folds_a = make_patient_specific_loso_edf(manifest_df, seed=3)
    folds_b = make_patient_specific_loso_edf(manifest_df, seed=3)
    assert [f.to_dict() for f in folds_a] == [f.to_dict() for f in folds_b]

    loso_a = make_zero_shot_loso_subject(manifest_df, seed=3)
    loso_b = make_zero_shot_loso_subject(manifest_df, seed=3)
    assert [f.to_dict() for f in loso_a] == [f.to_dict() for f in loso_b]

from __future__ import annotations

import pytest

from wearseizure.data.manifest import hash_manifest
from wearseizure.data.splits import (
    Fold,
    load_folds,
    make_patient_specific_loso_edf,
    make_zero_shot_loso_subject,
    save_folds,
    validate_fold,
)


def test_patient_specific_folds_have_no_intra_fold_overlap(synthetic_cohort):
    manifest_df, _ = synthetic_cohort
    folds = make_patient_specific_loso_edf(manifest_df, seed=0)
    assert len(folds) > 0
    for fold in folds:
        validate_fold(fold)  # raises on any overlap


def test_zero_shot_folds_have_no_intra_fold_overlap(synthetic_cohort):
    manifest_df, _ = synthetic_cohort
    folds = make_zero_shot_loso_subject(manifest_df, seed=0)
    assert len(folds) > 0
    for fold in folds:
        validate_fold(fold)


def test_zero_shot_held_out_subject_never_appears_in_train_or_val_of_its_own_fold(synthetic_cohort):
    manifest_df, _ = synthetic_cohort
    folds = make_zero_shot_loso_subject(manifest_df, seed=0)
    edf_to_subject = dict(zip(manifest_df["edf_id"], manifest_df["subject_id"]))

    for fold in folds:
        held_out_subject = fold.held_out_key
        for edf_id in fold.train_edf_ids | fold.val_edf_ids:
            assert edf_to_subject[edf_id] != held_out_subject
        for edf_id in fold.test_edf_ids:
            assert edf_to_subject[edf_id] == held_out_subject


def test_patient_specific_folds_only_use_edfs_from_the_same_subject(synthetic_cohort):
    manifest_df, _ = synthetic_cohort
    edf_to_subject = dict(zip(manifest_df["edf_id"], manifest_df["subject_id"]))
    folds = make_patient_specific_loso_edf(manifest_df, seed=0)

    for fold in folds:
        subject_id = edf_to_subject[fold.held_out_key]
        all_ids = fold.train_edf_ids | fold.val_edf_ids | fold.test_edf_ids
        assert all(edf_to_subject[e] == subject_id for e in all_ids)


def test_folds_carry_manifest_hash_and_reject_stale_manifest(synthetic_cohort, tmp_path):
    manifest_df, _ = synthetic_cohort
    folds = make_zero_shot_loso_subject(manifest_df, seed=0)
    path = tmp_path / "folds.json"
    save_folds(folds, str(path))

    reloaded = load_folds(str(path), expected_manifest_hash=hash_manifest(manifest_df))
    assert len(reloaded) == len(folds)

    with pytest.raises(ValueError, match="manifest_hash"):
        load_folds(str(path), expected_manifest_hash="not-the-real-hash")


def test_validate_fold_raises_on_synthetic_overlap():
    bad_fold = Fold(
        fold_id="bad",
        train_edf_ids=frozenset({"a", "b"}),
        val_edf_ids=frozenset({"b"}),  # overlaps train
        test_edf_ids=frozenset({"c"}),
        held_out_key="c",
        manifest_hash="deadbeef",
    )
    with pytest.raises(ValueError, match="overlap"):
        validate_fold(bad_fold)


def test_stale_folds_are_refused_when_the_manifest_changed(tmp_path, synthetic_cohort):
    """PROTOCOL.md calls splits version-locked by manifest hash. Until the
    scripts started passing `expected_manifest_hash`, nothing enforced it: a
    manifest rebuilt from changed data would pair silently with folds built
    from the old one, and every comparison against earlier results would be
    invalid with nothing to indicate it."""
    import pytest

    from wearseizure.data.manifest import hash_manifest
    from wearseizure.data.splits import load_folds, make_patient_specific_loso_edf, save_folds

    manifest_df, _ = synthetic_cohort
    folds = make_patient_specific_loso_edf(manifest_df, seed=0)
    path = tmp_path / "folds.json"
    save_folds(folds, str(path))

    # Matching hash: accepted, and identical to the unchecked load.
    good = hash_manifest(manifest_df)
    assert len(load_folds(str(path), expected_manifest_hash=good)) == len(folds)

    with pytest.raises(ValueError, match="Regenerate splits"):
        load_folds(str(path), expected_manifest_hash="0" * 64)


def test_dataset_build_does_not_hold_the_filtered_corpus_twice(synthetic_cohort):
    """Regression for the allocation that OOM-killed Phase 3.

    build_fold_datasets used to filter the train partition, then filter every
    partition again -- so train EDFs were filtered twice and both float64
    copies stayed alive. Together with two full-size temporaries inside
    fit_affine_normalizer, peak resident reached 72.6 GiB per process on the
    lever-L5 corpus against a check that had predicted 7.2.

    Asserting on real memory would be flaky, so this pins the structural
    property instead: every EDF is filtered exactly once.
    """
    from unittest.mock import patch

    from wearseizure.data.dataset import build_fold_datasets
    from wearseizure.data.splits import make_patient_specific_loso_edf
    from wearseizure.signal.filters import CausalBandpass

    manifest_df, records = synthetic_cohort
    fold = make_patient_specific_loso_edf(manifest_df, seed=0)[0]
    n_edfs = len(fold.train_edf_ids | fold.val_edf_ids | fold.test_edf_ids)

    real = CausalBandpass.apply_full_reset
    calls = []

    def counting(self, x):
        calls.append(len(x))
        return real(self, x)

    with patch.object(CausalBandpass, "apply_full_reset", counting):
        build_fold_datasets(records, fold, 4.0, 1.0)

    assert len(calls) == n_edfs, (
        f"filtered {len(calls)} times for {n_edfs} EDFs -- the duplicate pass is back"
    )

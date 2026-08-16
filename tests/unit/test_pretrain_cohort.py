"""Leakage safety of cohort pre-training (lever L1).

Cohort pre-training is the one place in this pipeline where a fold's model
sees data from outside its own patient, so the exclusion of the target patient
is the property that has to be pinned hardest.
"""
from __future__ import annotations

import pandas as pd
import pytest

from wearseizure.training.pretrain import cohort_pretrain_fold


def _manifest(n_subjects: int = 6, edfs_per_subject: int = 3) -> pd.DataFrame:
    rows = []
    for s in range(n_subjects):
        subject = f"chb{s:02d}"
        for e in range(edfs_per_subject):
            rows.append(
                {
                    "subject_id": subject,
                    "edf_id": f"{subject}_{e:02d}",
                    "channel": "P8-O2",
                    "fs_hz": 256,
                    "duration_s": 3600.0,
                    "seizure_events": "[]",
                }
            )
    return pd.DataFrame(rows)


def test_held_out_subject_never_appears_in_pretraining_data():
    df = _manifest()
    fold = cohort_pretrain_fold(df, held_out_subject="chb02", seed=0)

    held_out_ids = set(df.loc[df["subject_id"] == "chb02", "edf_id"])
    assert held_out_ids, "test fixture must actually contain the held-out subject"
    assert not (held_out_ids & fold.train_edf_ids)
    assert not (held_out_ids & fold.val_edf_ids)


def test_every_other_subject_is_used_and_train_val_are_disjoint():
    df = _manifest()
    fold = cohort_pretrain_fold(df, held_out_subject="chb02", seed=0)

    used = fold.train_edf_ids | fold.val_edf_ids
    expected = set(df.loc[df["subject_id"] != "chb02", "edf_id"])
    assert used == expected, "cohort pre-training should use all of the remaining corpus"
    assert not (fold.train_edf_ids & fold.val_edf_ids)


def test_val_split_is_at_subject_level_not_edf_level():
    # Early stopping during pre-training must be measured on patients the model
    # has not seen, otherwise it is measuring memorisation of known patients.
    df = _manifest()
    fold = cohort_pretrain_fold(df, held_out_subject="chb02", seed=0)

    subject_of = dict(zip(df["edf_id"], df["subject_id"]))
    train_subjects = {subject_of[e] for e in fold.train_edf_ids}
    val_subjects = {subject_of[e] for e in fold.val_edf_ids}
    assert val_subjects, "val partition must be non-empty"
    assert not (train_subjects & val_subjects)


def test_test_partition_is_empty():
    fold = cohort_pretrain_fold(_manifest(), held_out_subject="chb02", seed=0)
    assert fold.test_edf_ids == frozenset()


def test_is_deterministic_given_a_seed_and_varies_with_it():
    df = _manifest()
    a = cohort_pretrain_fold(df, held_out_subject="chb02", seed=0)
    b = cohort_pretrain_fold(df, held_out_subject="chb02", seed=0)
    assert a == b

    different_subject = cohort_pretrain_fold(df, held_out_subject="chb03", seed=0)
    assert different_subject.val_edf_ids != a.val_edf_ids or different_subject.train_edf_ids != a.train_edf_ids


def test_each_subject_gets_its_own_init_keyed_by_fold_id():
    df = _manifest()
    a = cohort_pretrain_fold(df, held_out_subject="chb02", seed=0)
    b = cohort_pretrain_fold(df, held_out_subject="chb03", seed=0)
    assert a.fold_id == "pretrain__chb02"
    assert b.fold_id == "pretrain__chb03"


def test_rejects_unknown_subject():
    with pytest.raises(ValueError):
        cohort_pretrain_fold(_manifest(), held_out_subject="chb99", seed=0)


def test_rejects_a_cohort_too_small_to_split():
    df = _manifest(n_subjects=2)
    with pytest.raises(ValueError, match="needs >=2 other subjects"):
        cohort_pretrain_fold(df, held_out_subject="chb00", seed=0)


def test_manifest_hash_travels_with_the_fold():
    df = _manifest()
    fold = cohort_pretrain_fold(df, held_out_subject="chb02", seed=0)
    assert fold.manifest_hash
    assert fold.held_out_key == "chb02"

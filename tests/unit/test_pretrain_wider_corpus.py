"""Leakage safety of the WIDER pre-training corpus (lever L5).

Lever L1 pre-trains each fold on the other 12 evaluation cases; L5 adds the
CHB-MIT cases the protocol does not evaluate on. That makes the corpus roughly
twice as large without touching what is scored -- but it also introduces a leak
that could not exist before: chb21 is chb01 recorded 1.5 years later, so an
id-string comparison is no longer enough to keep a patient out of their own
initialisation.

Everything here is about that boundary. The pre-training pool must never
contain a recording of the held-out person, under any case id.
"""
from __future__ import annotations

import pandas as pd
import pytest

from wearseizure.data.manifest import (
    hash_manifest,
    is_evaluation_case,
    subjects_sharing_identity,
)
from wearseizure.training.pretrain import build_cohort_manifest, cohort_pretrain_fold


def _rows(subject: str, n: int, prefix: str = "") -> list[dict]:
    return [
        {
            "subject_id": subject,
            "edf_id": f"{prefix}{subject}_{e:02d}",
            "channel_name": "P8-O2",
            "fs_hz": 256,
            "duration_sec": 3600.0,
            "seizure_events": "[]",
        }
        for e in range(n)
    ]


def _eval_manifest() -> pd.DataFrame:
    """The 13 evaluation cases, in the real Appendix A ids."""
    subjects = ["chb01", "chb02", "chb03", "chb04", "chb05", "chb07", "chb08",
                "chb10", "chb11", "chb15", "chb17", "chb22", "chb23"]
    return pd.DataFrame([r for s in subjects for r in _rows(s, 3)])


def _extra_manifest(subjects=("chb06", "chb21", "chb24")) -> pd.DataFrame:
    # "@channel" ids, matching what scripts/make_manifest.py emits for the
    # pre-training-only cases (one row per EDF per wearable position).
    return pd.DataFrame([r for s in subjects for r in _rows(s, 2, prefix="pre_")])


# ---------------------------------------------------------------------------
# The identity rule itself
# ---------------------------------------------------------------------------


def test_chb21_is_the_same_person_as_chb01():
    assert subjects_sharing_identity("chb01") == frozenset({"chb01", "chb21"})
    assert subjects_sharing_identity("chb21") == frozenset({"chb01", "chb21"})


def test_an_unrelated_case_is_only_itself():
    assert subjects_sharing_identity("chb06") == frozenset({"chb06"})


def test_only_the_thirteen_appendix_a_cases_count_as_evaluation_cases():
    assert is_evaluation_case("chb01")
    assert is_evaluation_case("chb23")
    assert not is_evaluation_case("chb06")
    assert not is_evaluation_case("chb21")


# ---------------------------------------------------------------------------
# The pre-training pool
# ---------------------------------------------------------------------------


def test_the_wider_corpus_reaches_the_pretraining_pool():
    eval_df, extra_df = _eval_manifest(), _extra_manifest()
    fold = cohort_pretrain_fold(eval_df, "chb02", seed=0, extra_manifest_df=extra_df)
    pool = fold.train_edf_ids | fold.val_edf_ids
    assert set(extra_df["edf_id"]) & pool, "lever L5 rows must actually be used"


def test_holding_out_chb01_also_excludes_chb21():
    eval_df, extra_df = _eval_manifest(), _extra_manifest()
    fold = cohort_pretrain_fold(eval_df, "chb01", seed=0, extra_manifest_df=extra_df)
    pool = fold.train_edf_ids | fold.val_edf_ids
    chb21_ids = set(extra_df.loc[extra_df["subject_id"] == "chb21", "edf_id"])
    assert chb21_ids, "test fixture must actually contain chb21"
    assert not (chb21_ids & pool), "chb21 is chb01; it must not train chb01's initialisation"
    # ...while an unrelated extra case is still available.
    chb06_ids = set(extra_df.loc[extra_df["subject_id"] == "chb06", "edf_id"])
    assert chb06_ids & pool


def test_holding_out_every_evaluation_case_never_leaks_that_person():
    eval_df, extra_df = _eval_manifest(), _extra_manifest()
    for subject in sorted(eval_df["subject_id"].unique()):
        fold = cohort_pretrain_fold(eval_df, subject, seed=0, extra_manifest_df=extra_df)
        pool = fold.train_edf_ids | fold.val_edf_ids
        identity = subjects_sharing_identity(subject)
        combined = build_cohort_manifest(eval_df, extra_df)
        own = set(combined.loc[combined["subject_id"].isin(identity), "edf_id"])
        assert not (own & pool), f"{subject}: leaked {sorted(own & pool)}"


def test_train_and_val_stay_disjoint_with_the_wider_corpus():
    fold = cohort_pretrain_fold(
        _eval_manifest(), "chb05", seed=0, extra_manifest_df=_extra_manifest()
    )
    assert not (fold.train_edf_ids & fold.val_edf_ids)
    assert fold.test_edf_ids == frozenset()


# ---------------------------------------------------------------------------
# Corpus bookkeeping
# ---------------------------------------------------------------------------


def test_overlapping_edf_ids_between_the_two_manifests_are_refused():
    eval_df = _eval_manifest()
    # A pre-training manifest that re-lists an evaluation recording would put
    # evaluation data into pre-training without any subject_id looking wrong.
    colliding = eval_df.head(2).copy()
    with pytest.raises(ValueError, match="disjoint"):
        build_cohort_manifest(eval_df, colliding)


def test_building_the_cohort_does_not_mutate_the_evaluation_manifest():
    eval_df = _eval_manifest()
    before = hash_manifest(eval_df)
    build_cohort_manifest(eval_df, _extra_manifest())
    assert hash_manifest(eval_df) == before, "evaluation manifest must be untouched"


def test_no_extra_manifest_is_a_no_op():
    eval_df = _eval_manifest()
    assert build_cohort_manifest(eval_df, None) is eval_df
    assert build_cohort_manifest(eval_df, pd.DataFrame()) is eval_df


def test_widening_the_corpus_changes_the_fold_manifest_hash():
    # This is what invalidates cached cohort initialisations. Without it,
    # turning L5 on would silently reuse the narrow init and look like L5 had
    # no effect.
    eval_df = _eval_manifest()
    narrow = cohort_pretrain_fold(eval_df, "chb02", seed=0)
    wide = cohort_pretrain_fold(eval_df, "chb02", seed=0, extra_manifest_df=_extra_manifest())
    assert narrow.manifest_hash != wide.manifest_hash

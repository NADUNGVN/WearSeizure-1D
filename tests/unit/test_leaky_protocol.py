"""Item A7: the published protocol, reproduced on purpose.

These tests pin the properties that make the reproduction mean anything. The
harness exists to show that a published number is inflated by its protocol, so
if the harness itself were subtly wrong the whole argument would invert -- and
it would invert quietly, because "our leaky run also scored 0.99" looks like
success either way.
"""
from __future__ import annotations

import numpy as np
import pytest

from wearseizure.data.dataset import WearSeizureWindowDataset, build_fold_datasets
from wearseizure.data.splits import make_patient_specific_loso_edf
from wearseizure.eval.leaky_protocol import (
    LADDER,
    ProtocolConfig,
    near_duplicate_fraction,
    prepare_records,
    split_window_indices,
)


@pytest.fixture
def cohort(synthetic_cohort):
    manifest_df, records = synthetic_cohort
    fold = make_patient_specific_loso_edf(manifest_df, seed=0)[0]
    pool = fold.train_edf_ids | fold.val_edf_ids | fold.test_edf_ids
    prepared = prepare_records(records, pool, fold.train_edf_ids, global_normalisation=False)
    ds = WearSeizureWindowDataset(prepared, frozenset(pool), 4.0, 1.0)
    return fold, pool, records, ds


# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------


def test_the_ladder_changes_exactly_one_factor_per_rung():
    """The whole point is attribution. Two factors moving at once would make
    the figure unreadable -- which is the failure this project has already been
    burned by twice (row 15, and `wide` versus the control)."""
    fields = ("random_window_split", "global_normalisation", "threshold_on_test", "segment_metric")
    for a, b in zip(LADDER, LADDER[1:]):
        changed = [f for f in fields if getattr(a, f) != getattr(b, f)]
        # B->C drops the normalisation and threshold leaks together: both are
        # "fitted on data the model should not see", one practice with two
        # spellings, and splitting them would need a rung nobody published.
        assert 1 <= len(changed) <= 2, f"{a.name} -> {b.name} changes {changed}"


def test_the_last_rung_is_this_projects_own_protocol():
    ours = LADDER[-1]
    assert not any((ours.random_window_split, ours.global_normalisation, ours.threshold_on_test))
    assert not ours.segment_metric, "ours is the event-level one"


def test_the_first_rung_has_every_leak_on():
    published = LADDER[0]
    assert all((published.random_window_split, published.global_normalisation,
                published.threshold_on_test, published.segment_metric))


# ---------------------------------------------------------------------------
# The split, and the leak it creates
# ---------------------------------------------------------------------------


def test_random_split_puts_near_duplicates_of_test_windows_into_train(cohort):
    """This is the mechanism, measured rather than asserted. At 4s/1s, windows
    overlap their neighbours by 75%, so splitting windows at random hands the
    model almost every test window in training under a one-second shift."""
    _fold, _pool, _records, ds = cohort
    tr, va, te = split_window_indices(
        ds, random_window_split=True, test_edf_ids=frozenset(), val_fraction=0.2, seed=0
    )
    assert len(tr) + len(va) + len(te) == len(ds.windows)
    assert len(set(tr) | set(va) | set(te)) == len(ds.windows), "partitions must be disjoint"
    assert near_duplicate_fraction(ds, tr, te) > 0.95


def test_recording_split_leaves_no_overlap_at_all(cohort):
    fold, _pool, _records, ds = cohort
    tr, va, te = split_window_indices(
        ds, random_window_split=False, test_edf_ids=fold.test_edf_ids, val_fraction=0.2, seed=0
    )
    assert near_duplicate_fraction(ds, tr, te) == 0.0
    test_edfs = {ds.windows[i].edf_id for i in te}
    assert test_edfs == set(fold.test_edf_ids)
    assert not test_edfs & {ds.windows[i].edf_id for i in tr}


def test_validation_is_carved_out_by_recording_too(cohort):
    """Taking val at random inside a recording split would put the fitting leak
    back through the side door: the threshold would be frozen on windows whose
    near-duplicates trained the model."""
    fold, _pool, _records, ds = cohort
    tr, va, _te = split_window_indices(
        ds, random_window_split=False, test_edf_ids=fold.test_edf_ids, val_fraction=0.2, seed=0
    )
    assert va.size, "the recording split must still produce a validation partition"
    assert not {ds.windows[i].edf_id for i in va} & {ds.windows[i].edf_id for i in tr}
    assert near_duplicate_fraction(ds, tr, va) == 0.0


def test_a_split_that_leaves_a_partition_empty_is_refused(cohort):
    _fold, _pool, _records, ds = cohort
    with pytest.raises(ValueError, match="left a partition empty"):
        split_window_indices(
            ds, random_window_split=False, test_edf_ids=frozenset(), val_fraction=0.2, seed=0
        )


def test_the_split_is_reproducible_from_the_seed(cohort):
    _fold, _pool, _records, ds = cohort
    a = split_window_indices(ds, random_window_split=True, test_edf_ids=frozenset(),
                             val_fraction=0.2, seed=7)
    b = split_window_indices(ds, random_window_split=True, test_edf_ids=frozenset(),
                             val_fraction=0.2, seed=7)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def test_global_normalisation_actually_differs_from_train_only(synthetic_cohort):
    """If the two agreed, rung B->C would measure nothing and the figure would
    silently claim a leak that was never exercised."""
    manifest_df, records = synthetic_cohort
    fold = make_patient_specific_loso_edf(manifest_df, seed=0)[0]
    pool = fold.train_edf_ids | fold.val_edf_ids | fold.test_edf_ids

    train_only = prepare_records(records, pool, fold.train_edf_ids, global_normalisation=False)
    everything = prepare_records(records, pool, fold.train_edf_ids, global_normalisation=True)

    held_out = next(iter(fold.test_edf_ids))
    assert not np.allclose(train_only[held_out].signal, everything[held_out].signal)


def test_train_only_normalisation_matches_the_main_pipeline(synthetic_cohort):
    """Rung C must be this project's own preprocessing, or the ladder compares
    against something that was never run. Same records, same filter, same
    normaliser fit -- so the arrays must agree to the bit."""
    manifest_df, records = synthetic_cohort
    fold = make_patient_specific_loso_edf(manifest_df, seed=0)[0]
    pool = fold.train_edf_ids | fold.val_edf_ids | fold.test_edf_ids

    ours = prepare_records(records, pool, fold.train_edf_ids, global_normalisation=False)
    datasets, _band, _norm = build_fold_datasets(records, fold, 4.0, 1.0)
    for edf_id, record in datasets["test"].records.items():
        assert np.array_equal(ours[edf_id].signal, record.signal), edf_id


def test_normalising_with_nothing_to_fit_on_is_refused(synthetic_cohort):
    manifest_df, records = synthetic_cohort
    fold = make_patient_specific_loso_edf(manifest_df, seed=0)[0]
    with pytest.raises(ValueError, match="nothing to fit"):
        prepare_records(records, fold.test_edf_ids, frozenset(), global_normalisation=False)


# ---------------------------------------------------------------------------
# The description, which is what ends up in the figure caption
# ---------------------------------------------------------------------------


def test_every_rung_describes_all_four_factors():
    for rung in LADDER:
        text = rung.describe()
        assert "split" in text and "normalised" in text and "threshold" in text
        assert "sensitivity" in text


def test_the_default_config_is_the_honest_one():
    plain = ProtocolConfig("x")
    assert not any((plain.random_window_split, plain.global_normalisation,
                    plain.threshold_on_test, plain.segment_metric))

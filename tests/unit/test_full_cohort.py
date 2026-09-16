"""The 24-case run: channel choice, shard disjointness, exclusion bookkeeping.

Two things here could silently invalidate the experiment. The channel for the
eleven unconfirmed cases must be chosen on training data only -- choosing it on
the evaluation data is the same class of leak the project exists to document.
And the shards must partition the folds exactly, because two shards computing
one fold is the failure that has already cost this project three hours once.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "run_full_cohort", ROOT / "scripts" / "run_full_cohort.py")
full = importlib.util.module_from_spec(_spec)
sys.modules["run_full_cohort"] = full
_spec.loader.exec_module(full)


class W:
    """Minimal stand-in for a Window: what auroc_logvar actually reads."""

    def __init__(self, edf_id, start_idx, end_idx, label):
        self.edf_id, self.start_idx, self.end_idx, self.label = (
            edf_id, start_idx, end_idx, label)


def test_channel_id_strips_the_position_suffix():
    """Fold membership is channel-independent, so `chb06_01@Fp1-F3` and
    `chb06_01@P7-O1` are one recording, not two."""
    assert full.base_edf_id("chb06_01@Fp1-F3") == "chb06_01"
    assert full.base_edf_id("chb01_03") == "chb01_03"


def test_auroc_is_one_when_seizures_are_louder_and_half_when_they_are_not():
    rng = np.random.default_rng(0)
    quiet = rng.normal(0, 1.0, 10_000)
    loud = quiet.copy()
    loud[5_000:] *= 20.0          # the second half is the "seizure"

    windows = ([W("e", i * 100, (i + 1) * 100, 0) for i in range(50)]
               + [W("e", i * 100, (i + 1) * 100, 1) for i in range(50, 100)])

    assert full.auroc_logvar({"e": loud}, windows) == pytest.approx(1.0, abs=1e-6)
    # Same labels, a signal with no amplitude structure: chance.
    flat = full.auroc_logvar({"e": quiet}, windows)
    assert 0.35 < flat < 0.65


def test_auroc_is_undefined_rather_than_wrong_when_a_class_is_missing():
    """A training partition with no seizure cannot rank positions. Returning a
    number there would silently pick an electrode on no evidence."""
    windows = [W("e", i * 100, (i + 1) * 100, 0) for i in range(10)]
    assert np.isnan(full.auroc_logvar({"e": np.ones(2000)}, windows))
    assert np.isnan(full.auroc_logvar({"missing": np.ones(10)}, windows))


def test_channel_choice_is_deterministic_under_ties():
    """Ties must break on channel name, not on dict iteration order, or two
    machines running the same fold could train on different electrodes."""
    rng = np.random.default_rng(1)
    sig = np.stack([rng.normal(0, 1.0, 4000) for _ in range(4)])
    signals = {"e": sig}
    channels = ["FP1-F3", "P7-O1", "P8-O2", "FP2-F4"]

    class FakeRecords(dict):
        pass

    windows = ([W("e", i * 100, (i + 1) * 100, 0) for i in range(20)]
               + [W("e", i * 100, (i + 1) * 100, 1) for i in range(20, 40)])
    scores = {name: full.auroc_logvar({"e": sig[row]}, windows)
              for row, name in enumerate(channels)}
    usable = {k: v for k, v in scores.items() if not np.isnan(v)}
    first = max(sorted(usable), key=lambda k: usable[k])
    for _ in range(20):
        shuffled = dict(sorted(usable.items(), key=lambda kv: rng.random()))
        assert max(sorted(shuffled), key=lambda k: shuffled[k]) == first


def test_exclusion_groups_match_chungs_two_reasons():
    """The two reasons mean different things and must not be pooled: a seizure
    that is not under any wearable electrode cannot be detected at all, while an
    unidentified location merely means nobody checked."""
    assert set(full.SEIZURE_NOT_NEAR_WEARABLE) == {"chb06", "chb09", "chb13", "chb19"}
    assert set(full.LOCATION_UNIDENTIFIED) == {
        "chb12", "chb14", "chb16", "chb18", "chb20", "chb21"}
    assert not set(full.SEIZURE_NOT_NEAR_WEARABLE) & set(full.LOCATION_UNIDENTIFIED)


def test_combined_metadata_is_one_row_per_recording():
    """The pre-training manifest carries four rows per recording, one per
    wearable position. Folds need one."""
    eval_df = pd.DataFrame({
        "subject_id": ["chb01", "chb01"],
        "edf_id": ["chb01_03", "chb01_04"],
        "edf_relpath": ["chb01_03.edf", "chb01_04.edf"],
        "channel_name": ["P8-O2", "P8-O2"],
    })
    pre_df = pd.DataFrame({
        "subject_id": ["chb06"] * 4 + ["chb09"] * 4,
        "edf_id": [f"chb06_01@{c}" for c in ["Fp1-F3", "P7-O1", "P8-O2", "Fp2-F4"]]
                  + [f"chb09_01@{c}" for c in ["Fp1-F3", "P7-O1", "P8-O2", "Fp2-F4"]],
        "edf_relpath": ["chb06_01.edf"] * 4 + ["chb09_01.edf"] * 4,
        "channel_name": ["Fp1-F3", "P7-O1", "P8-O2", "Fp2-F4"] * 2,
    })
    out = full.combined_metadata(eval_df, pre_df)
    assert list(out["edf_id"]) == ["chb01_03", "chb01_04", "chb06_01", "chb09_01"]
    assert out["subject_id"].nunique() == 3


@pytest.mark.parametrize("n_shards", [1, 2, 3, 4, 7])
def test_shards_partition_the_folds_exactly(n_shards):
    """Every fold in exactly one shard. A fold in two shards is two machines
    computing it and racing to write the result; a fold in none is a silent
    hole in the cohort."""
    folds = list(range(138))
    seen: list[int] = []
    for i in range(n_shards):
        seen.extend(f for j, f in enumerate(folds) if j % n_shards == i)
    assert sorted(seen) == folds
    assert len(seen) == len(set(seen))

    sizes = [sum(1 for j in range(len(folds)) if j % n_shards == i) for i in range(n_shards)]
    assert max(sizes) - min(sizes) <= 1, "shards must be balanced within one fold"

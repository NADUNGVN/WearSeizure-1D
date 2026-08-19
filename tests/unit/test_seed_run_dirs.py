"""Multi-seed run layout (lever L7).

`train.seeds` was declared from the first commit and read by nothing, so none
of the 26 runs in `docs/EXPERIMENT_LOG_G1a.md` carries an error bar. These
tests pin the two properties that make turning it on safe: a single-seed run
still means exactly `cfg.seed`, and two seeds can never write over each other's
per-fold metrics.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from wearseizure.utils.paths import (
    fold_run_dir,
    pretrain_cache_dir,
    seeds_from_cfg,
    warn_if_legacy_artifacts,
)


def _cfg(seed: int = 0, seeds=None):
    return OmegaConf.create({"seed": seed, "train": {"seeds": seeds}})


def test_default_is_a_single_seed_taken_from_cfg_seed():
    assert seeds_from_cfg(_cfg(seed=0)) == [0]
    assert seeds_from_cfg(_cfg(seed=7)) == [7]


def test_explicit_seed_list_is_honoured():
    assert seeds_from_cfg(_cfg(seeds=[0, 1, 2])) == [0, 1, 2]


def test_empty_or_duplicated_seed_lists_are_refused():
    # An empty list is almost certainly a typo for null, and silently training
    # nothing is the worst possible response to it.
    with pytest.raises(ValueError, match="empty list"):
        seeds_from_cfg(_cfg(seeds=[]))
    # Duplicates would resolve to the same directory and the second pass would
    # "resume" from the first, quietly producing 2 identical "seeds".
    with pytest.raises(ValueError, match="duplicates"):
        seeds_from_cfg(_cfg(seeds=[0, 1, 0]))


def test_each_seed_gets_its_own_directory():
    dirs = {fold_run_dir("/art", "wearseizure1d", "loso", "w4s", s) for s in (0, 1, 2)}
    assert len(dirs) == 3
    assert fold_run_dir("/art", "m", "s", "w", 1).name == "seed1"


def test_pretrain_cache_is_keyed_by_seed_too():
    # cohort_pretrain_fold draws its subject-level validation split from
    # rng_for(..., base_seed=seed), so two seeds are two different cohort
    # initialisations and must not share a cache entry.
    a = pretrain_cache_dir("/art", "m", "w4s", 0)
    b = pretrain_cache_dir("/art", "m", "w4s", 1)
    assert a != b


def test_legacy_artifacts_one_level_up_are_reported(tmp_path: Path, caplog):
    window_dir = tmp_path / "wearseizure1d" / "loso" / "w4s"
    window_dir.mkdir(parents=True)
    (window_dir / "chb01__chb01_03.metrics.json").write_text("{}", encoding="utf-8")
    run_dir = window_dir / "seed0"

    log = logging.getLogger("test_legacy")
    with caplog.at_level(logging.WARNING, logger="test_legacy"):
        warn_if_legacy_artifacts(run_dir, log)
    assert "before the seed<N> directory level" in caplog.text
    assert "mv" in caplog.text, "the warning must say how to adopt them"


def test_no_warning_once_the_seed_directory_has_its_own_artifacts(tmp_path: Path, caplog):
    window_dir = tmp_path / "m" / "s" / "w"
    run_dir = window_dir / "seed0"
    run_dir.mkdir(parents=True)
    (window_dir / "old.metrics.json").write_text("{}", encoding="utf-8")
    (run_dir / "chb01__chb01_03.metrics.json").write_text("{}", encoding="utf-8")

    log = logging.getLogger("test_legacy_quiet")
    with caplog.at_level(logging.WARNING, logger="test_legacy_quiet"):
        warn_if_legacy_artifacts(run_dir, log)
    assert caplog.text == ""


def test_no_warning_when_there_is_nothing_to_migrate(tmp_path: Path, caplog):
    run_dir = tmp_path / "m" / "s" / "w" / "seed0"
    run_dir.mkdir(parents=True)
    log = logging.getLogger("test_legacy_none")
    with caplog.at_level(logging.WARNING, logger="test_legacy_none"):
        warn_if_legacy_artifacts(run_dir, log)
    assert caplog.text == ""

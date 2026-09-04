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
    run_tag_from_cfg,
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


def test_run_tag_separates_runs_that_the_path_would_otherwise_collide(tmp_path: Path):
    # Lever L5 changes the pre-training corpus, which the path does not encode,
    # so without a tag an L5 run overwrites the run it exists to be compared
    # against -- silently destroying rows 32-34.
    plain = fold_run_dir("/art", "m", "loso", "w4s", 0)
    tagged = fold_run_dir("/art", "m", "loso", "w4s", 0, "L5")
    assert plain != tagged
    assert tagged.parent.name == "w4s__L5"
    assert pretrain_cache_dir("/art", "m", "w4s", 0) != pretrain_cache_dir("/art", "m", "w4s", 0, "L5")


def test_no_tag_leaves_every_existing_path_unchanged():
    assert fold_run_dir("/art", "m", "loso", "w4s", 0, "") == fold_run_dir("/art", "m", "loso", "w4s", 0)
    assert pretrain_cache_dir("/art", "m", "w4s", 0, "") == pretrain_cache_dir("/art", "m", "w4s", 0)


def test_run_tag_is_read_from_config_and_validated():
    assert run_tag_from_cfg(OmegaConf.create({"train": {"run_tag": None}})) == ""
    assert run_tag_from_cfg(OmegaConf.create({"train": {"run_tag": "L5"}})) == "L5"
    assert run_tag_from_cfg(OmegaConf.create({"train": {"run_tag": "  L5  "}})) == "L5"
    # A tag becomes a directory name, so anything that could escape it is refused.
    for bad in ("L5/../evil", "a b", "x;y"):
        with pytest.raises(ValueError, match="alphanumeric"):
            run_tag_from_cfg(OmegaConf.create({"train": {"run_tag": bad}}))


def test_every_fold_run_dir_call_in_the_scripts_passes_a_tag():
    """Every `fold_run_dir(...)` call site must pass the run tag.

    One did not: the multi-seed summary path in scripts/evaluate.py. Per-seed
    reports went to the tagged directory while `report_multiseed.json` went to
    the untagged one, so a lever-L5 run overwrote the control arm's summary with
    its own numbers. Nothing errored; the control simply changed underneath a
    comparison table.
    """
    import ast
    import inspect

    from wearseizure.utils import paths

    # Parsed, not grepped. The first version searched the call text for the
    # string "run_tag", which a call can satisfy while being wrong:
    # fold_run_dir(..., run_tag=run_tag) contains the string but raises at
    # runtime, because the parameter is named `tag`. A guard that a broken call
    # can satisfy is worse than no guard, because it is trusted.
    tag_index = list(inspect.signature(paths.fold_run_dir).parameters).index("tag")

    scripts = Path(__file__).resolve().parents[2] / "scripts"
    offenders = []
    for path in scripts.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "fold_run_dir"):
                continue
            if not (len(node.args) > tag_index or any(k.arg == "tag" for k in node.keywords)):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, "fold_run_dir called without its tag argument at: " + ", ".join(offenders)


def test_pretrain_cache_sharing_is_opt_in_and_lives_under_pretrain():
    """`share_cache_with_control` decides whether a tagged run reuses the
    control's cohort initialisations.

    It has to be read from `train.pretrain`, which is where train.py looks. An
    earlier edit appended it to the end of the config file, where YAML folded it
    into the `distill:` block instead -- the key existed, the docs described it,
    and it could never have taken effect.

    And it has to default to false: turning it on for a lever that DOES change
    pre-training (as L4 does) would silently reuse the control initialisations,
    and the experiment would measure nothing while looking like it ran.
    """
    import yaml

    cfg_path = Path(__file__).resolve().parents[2] / "configs" / "train" / "default.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert cfg["pretrain"]["share_cache_with_control"] is False
    assert "share_cache_with_control" not in cfg["distill"]


def test_sharing_the_cache_selects_the_untagged_directory():
    tagged = pretrain_cache_dir("/art", "m", "w4s", 0, "L3")
    shared = pretrain_cache_dir("/art", "m", "w4s", 0, "")
    assert tagged != shared
    assert shared == pretrain_cache_dir("/art", "m", "w4s", 0)

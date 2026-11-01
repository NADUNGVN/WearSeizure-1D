"""Inference-time seed ensemble (Phase 9).

The ensemble is evaluated through the same `evaluate_fold` path as a single
model, so what needs pinning is the wrapper itself and the two ways a run
could silently measure the wrong thing: members missing for some folds, and
two different ensembles writing to one directory.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from wearseizure.models.ensemble import LogitMeanEnsemble, ensemble_tag, missing_member_checkpoints
from wearseizure.models.wearseizure1d import WearSeizure1D


def _model(seed: int) -> nn.Module:
    torch.manual_seed(seed)
    m = WearSeizure1D(in_channels=1, input_len=1024, stem_out_channels=8,
                      stage_out_channels=(16, 24, 32, 48), context_channels=64,
                      dilations=(1, 2, 4), num_classes=2, kernel_mode="k5_only")
    return m.eval()


def test_output_is_the_mean_of_member_logits():
    members = [_model(s) for s in (0, 1, 2)]
    x = torch.randn(5, 1, 1024)
    with torch.no_grad():
        expected = (members[0](x) + members[1](x) + members[2](x)) / 3
        got = LogitMeanEnsemble(members).eval()(x)
    assert got.shape == (5, 2)
    torch.testing.assert_close(got, expected)


def test_identical_members_reproduce_the_single_model():
    m = _model(0)
    twin = _model(0)
    x = torch.randn(3, 1, 1024)
    with torch.no_grad():
        torch.testing.assert_close(LogitMeanEnsemble([m, twin]).eval()(x), m(x))


def test_device_move_reaches_every_member():
    ens = LogitMeanEnsemble([_model(0), _model(1)])
    ens.to(torch.float64)
    assert all(p.dtype == torch.float64 for p in ens.parameters())
    assert sum(1 for _ in ens.parameters()) == 2 * sum(1 for _ in _model(0).parameters())


def test_one_member_is_not_an_ensemble():
    with pytest.raises(ValueError):
        LogitMeanEnsemble([_model(0)])


def test_tag_encodes_the_members():
    assert ensemble_tag("L8", [0, 1, 2]) == "L8ens012"
    assert ensemble_tag("L8", [0, 2]) != ensemble_tag("L8", [0, 1])


@pytest.mark.parametrize("seeds", [[0], [1, 1], [1, 12]])
def test_tag_refuses_ambiguous_or_degenerate_seed_sets(seeds):
    with pytest.raises(ValueError):
        ensemble_tag("L8", seeds)


def test_missing_checkpoints_are_listed_per_member(tmp_path: Path):
    dirs = [tmp_path / f"seed{s}" for s in (0, 1, 2)]
    for d in dirs:
        d.mkdir()
    folds = ["chb01__chb01_03", "chb01__chb01_04"]
    for d in dirs:
        for f in folds:
            (d / f"{f}.pt").write_bytes(b"x")
    assert missing_member_checkpoints(dirs, folds) == []

    (dirs[1] / f"{folds[0]}.pt").unlink()
    assert missing_member_checkpoints(dirs, folds) == [dirs[1] / f"{folds[0]}.pt"]


def test_a_directory_is_not_a_checkpoint(tmp_path: Path):
    d = tmp_path / "seed0"
    (d / "chb01__chb01_03.pt").mkdir(parents=True)
    assert missing_member_checkpoints([d], ["chb01__chb01_03"]) == [d / "chb01__chb01_03.pt"]

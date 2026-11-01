"""Inference-time seed ensemble: one prediction from several trained seeds.

Why this exists
---------------
The headline configuration (L8, three seeds) varies by seed far more than by
any lever measured since: fold-averaged sensitivity is 0.9242, 0.9394 and
0.9848 for seeds 1, 2 and 0, and one to five of the 66 folds miss their
seizure entirely depending on the draw. Averaging the members' logits is the
standard remedy for exactly that kind of variance, and it is cheap on the
target: the accelerator is programmable, so three weight sets run back to back
on the same datapath, three times ~0.2 ms per one-second hop.

What it is not
--------------
Not a new training run, and not a free lunch to be read off a point estimate:
an ensemble of three costs three times the MACs and the weight memory of one
member, and whether it beats a single seed is a question for the paired
bootstrap against the three-seed mean, not for the best seed.

Logits rather than probabilities are averaged because that is what the
hardware path can do without a softmax: the host already receives integer
logits per member and can add them.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn


class LogitMeanEnsemble(nn.Module):
    """Mean of the members' logits, so the output is a drop-in for one model."""

    def __init__(self, members: Sequence[nn.Module]) -> None:
        super().__init__()
        if len(members) < 2:
            raise ValueError(f"an ensemble needs at least two members, got {len(members)}")
        self.members = nn.ModuleList(members)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([m(x) for m in self.members], dim=0).mean(dim=0)


def ensemble_tag(source_tag: str, seeds: Sequence[int]) -> str:
    """Run tag the ensemble writes under, derived from what it is built from.

    `L8` with seeds 0, 1, 2 becomes `L8ens012`. Encoding the members in the tag
    means two ensembles over different seed sets can never land in the same
    directory and overwrite each other.
    """
    if len(seeds) < 2:
        raise ValueError(f"an ensemble needs at least two seeds, got {list(seeds)}")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"duplicate seeds in ensemble: {list(seeds)}")
    if any(s < 0 or s > 9 for s in seeds):
        # One digit per seed keeps the tag unambiguous: 1,12 and 11,2 would
        # otherwise both read "112".
        raise ValueError(f"ensemble seeds must be 0-9 to keep the tag unambiguous: {list(seeds)}")
    return f"{source_tag}ens{''.join(str(s) for s in seeds)}"


def missing_member_checkpoints(member_dirs: Sequence[Path], fold_ids: Sequence[str]) -> list[Path]:
    """Every member checkpoint that should exist and does not.

    The caller refuses to run while this is non-empty. Silently ensembling the
    members that happen to be present would score some folds with three models
    and others with two, and report the mixture as one configuration.
    """
    return [d / f"{fid}.pt" for d in member_dirs for fid in fold_ids if not (d / f"{fid}.pt").is_file()]

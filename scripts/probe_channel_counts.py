"""How many channels does each fold actually share? Read labels, not samples.

    python scripts/probe_channel_counts.py profile=server data=chbmit

A multi-channel experiment needs one number before it can be designed: how many
channels a model would read. CHB-MIT does not supply a single answer. The
montage is not uniform -- chb04 has five files with 23 channels and thirty-five
with 24, some files carry a duplicated label, and a few use a different montage
entirely -- so "23 channels" is a property of a file, not of the dataset.

What a fold can offer is the INTERSECTION of channel names across every one of
its recordings. Anything outside that intersection is missing from at least one
recording the model would have to run on, and zero-filling it would train on
fabricated signal.

So this reports, per fold, how many channels survive that intersection, and
whether the count is the same across folds. If it is not, `in_channels` differs
per fold and every downstream shape does too -- which is a design constraint
worth knowing before any GPU time is spent, not after.

Reads only EDF headers, so it costs seconds rather than hours.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import hydra
from omegaconf import DictConfig

from wearseizure.data.manifest import hash_manifest, load_manifest
from wearseizure.data.splits import load_folds
from wearseizure.training.distill import MIN_TEACHER_CHANNELS, fold_common_channels
from wearseizure.utils.env import bootstrap_env
from wearseizure.utils.logging import get_logger
from wearseizure.utils.profile_guard import check_profile_data_pairing

log = get_logger(__name__)
bootstrap_env(sys.argv)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    check_profile_data_pairing(cfg)
    manifest_df = load_manifest(str(Path(cfg.data.manifest_path)))
    folds = load_folds(str(Path(cfg.split.folds_path)),
                       expected_manifest_hash=hash_manifest(manifest_df))
    raw_dir = str(cfg.data.raw_dir)

    counts: Counter[int] = Counter()
    per_fold: list[tuple[str, int]] = []
    refused: list[tuple[str, str]] = []
    everywhere: set[str] | None = None

    for fold in folds:
        edf_ids = fold.train_edf_ids | fold.val_edf_ids | fold.test_edf_ids
        try:
            common = fold_common_channels(manifest_df, raw_dir, edf_ids)
        except ValueError as exc:
            # Below MIN_TEACHER_CHANNELS the helper refuses rather than quietly
            # testing a narrower hypothesis. A refused fold is a real result:
            # it is a fold no multi-channel model could run on.
            refused.append((fold.fold_id, str(exc).split(".")[0]))
            continue
        counts[len(common)] += 1
        per_fold.append((fold.fold_id, len(common)))
        everywhere = set(common) if everywhere is None else everywhere & set(common)

    print(f"\n{len(per_fold)} of {len(folds)} folds have >= {MIN_TEACHER_CHANNELS} "
          "channels common to all their recordings\n")
    print("channels shared within a fold:")
    for n, k in sorted(counts.items()):
        print(f"  {n:3d} channels   {k:3d} folds")

    if refused:
        print(f"\n{len(refused)} folds refused:")
        for fold_id, why in refused[:10]:
            print(f"  {fold_id}: {why}")

    if per_fold:
        lo = min(per_fold, key=lambda t: t[1])
        hi = max(per_fold, key=lambda t: t[1])
        print(f"\nfewest : {lo[1]} channels ({lo[0]})")
        print(f"most   : {hi[1]} channels ({hi[0]})")

    # The intersection across every fold: the only channel set a SINGLE
    # architecture could read everywhere. If it is much smaller than the
    # per-fold counts, one model per fold is the only option -- which the
    # patient-specific protocol already implies, but the shapes then differ
    # fold to fold and nothing about the network is shared.
    if everywhere is not None:
        print(f"\ncommon to EVERY fold: {len(everywhere)} channels")
        if everywhere:
            print("  " + ", ".join(sorted(everywhere)))

    if len(counts) > 1:
        print("\nThe count is NOT uniform, so `in_channels` differs per fold and so "
              "does every\nlayer shape after it. A multi-channel run therefore builds "
              "a differently sized\nmodel per fold, and its parameter count is a range "
              "rather than a number.")


if __name__ == "__main__":
    main()

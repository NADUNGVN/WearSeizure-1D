"""Report the teacher montage every lever-L3 fold would get. Read-only.

Phase 5 aborted at fold 17 of 66 because chb04's EDFs carry either 23 or 24
channels. `fold_common_channels` now intersects by name instead of demanding
equal counts, and this script answers the question that decides whether that is
enough before another multi-day run is committed to it: how many channels does
each fold actually have in common, and is any fold left too thin for
"multi-channel teacher" to still mean anything?

Reads EDF headers only -- no sample data, no GPU, no writes.

    python scripts/check_fold_montage.py profile=server data=chbmit

Exits non-zero if any fold falls under MIN_TEACHER_CHANNELS, so it can gate a
run rather than merely inform one.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import hydra
from omegaconf import DictConfig

from wearseizure.data.io_edf import edf_channel_labels
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
    if cfg.data.name == "synthetic":
        # The synthetic cohort is single-channel .npy by construction, so there
        # is no montage to intersect and a run here would prove nothing about
        # the real one. `tests/unit/test_distillation.py` covers the logic.
        log.error("this check needs real multi-channel EDFs; run it with data=chbmit")
        sys.exit(2)
    folds = load_folds(str(Path(cfg.split.folds_path)), expected_manifest_hash=hash_manifest(manifest_df))
    raw_dir = cfg.data.raw_dir

    sizes: list[int] = []
    per_subject: dict[str, set[int]] = {}
    thin: list[str] = []
    for fold in folds:
        edf_ids = fold.train_edf_ids | fold.val_edf_ids
        # The per-file counts are what the old equal-counts rule tripped over,
        # so report them next to the intersection rather than only the verdict.
        counts = Counter(
            sum(
                1 for lbl in edf_channel_labels(
                    str(Path(raw_dir) / row["subject_id"] / row["edf_relpath"])
                ) if lbl.strip() and lbl.strip().upper() not in ("-", "--")
            )
            for _, row in manifest_df[manifest_df["edf_id"].isin(edf_ids)].iterrows()
        )
        try:
            n = len(fold_common_channels(manifest_df, raw_dir, edf_ids))
        except ValueError as exc:
            thin.append(f"{fold.fold_id}: {exc}")
            log.error(f"{fold.fold_id:<28} per-file counts {dict(counts)}  -> REFUSED: {exc}")
            continue
        sizes.append(n)
        per_subject.setdefault(fold.fold_id.split("__")[0], set()).add(n)
        uniform = "" if len(counts) == 1 else "  (files disagree)"
        log.info(f"{fold.fold_id:<28} per-file counts {dict(counts)}  -> common {n}{uniform}")

    log.info("=" * 72)
    for subject in sorted(per_subject):
        log.info(f"{subject}: montage sizes across its folds {sorted(per_subject[subject])}")
    if sizes:
        log.info(
            f"{len(sizes)}/{len(folds)} folds usable; common channels "
            f"min={min(sizes)} median={sorted(sizes)[len(sizes) // 2]} max={max(sizes)}"
        )
    if thin:
        log.error(f"{len(thin)} fold(s) under the {MIN_TEACHER_CHANNELS}-channel floor:")
        for line in thin:
            log.error(f"  {line}")
        sys.exit(1)
    log.info(f"every fold clears the {MIN_TEACHER_CHANNELS}-channel floor; L3 can run on all of them")


if __name__ == "__main__":
    main()

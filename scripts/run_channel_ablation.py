"""What does the single-channel constraint actually cost? Measure it.

    python scripts/run_channel_ablation.py profile=server data=chbmit \
        model=wearseizure1d_k5only +ablation.arms=[1,4,18] 'train.seeds=[0]'

Chung et al. 2024 built three detectors on CHB-MIT -- 18-channel, 4-channel and
single-channel -- and reported that segment-level sensitivity falls as channels
are removed: 98.66% -> 97.31% -> 96.76%. That is the number a reviewer will
quote back when asked why this project uses one channel.

But it was measured under their protocol, whose segment-level stage pools
overlapping 4-second windows from every recording and splits them 7:2:1 at
random. This project has already measured what that split is worth on this
data: 31 percentage points of window sensitivity. So the published 1.9-point
channel penalty is entangled with a 31-point protocol effect, and nobody has
measured what one channel costs when the protocol is clean.

That is what this script measures. Three arms, identical in every respect
except how many channels the model reads:

    1 channel   each patient's clinically confirmed wearable position
    4 channels  Chung et al.'s wearable subset
    18 channels Chung et al.'s full montage

Same folds, same seeds, same window, same causal filtering, same
threshold-on-validation discipline, same leave-one-seizure-recording-out split.
The single-channel arm is re-run here rather than reused from the production
results, because an arm that went through different code is not a control.

Results land in
    <artifacts>/channel_ablation/<n>ch/seed<N>/<fold_id>.json
one file per fold, and a fold whose file exists is skipped -- so an interrupted
run continues where it stopped.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader

from wearseizure.data.loader import load_records_from_manifest
from wearseizure.data.manifest import hash_manifest, load_manifest, subject_to_channel
from wearseizure.data.sampler import make_class_balanced_sampler
from wearseizure.data.splits import load_folds, subject_from_fold_id
from wearseizure.models.factory import build_model
from wearseizure.training.distill import (
    MultiChannelWindowDataset,
    fold_common_channels,
    prepare_multichannel_signals,
    windows_for_partition,
)
from wearseizure.training.engine_baseline import evaluate_fold
from wearseizure.training.loop import train_classifier
from wearseizure.utils.env import bootstrap_env
from wearseizure.utils.logging import get_logger
from wearseizure.utils.paths import ensure_dir, fold_run_dir, run_tag_from_cfg, seeds_from_cfg
from wearseizure.utils.profile_guard import check_profile_data_pairing
from wearseizure.utils.seeding import seed_everything

log = get_logger(__name__)
bootstrap_env(sys.argv)

# Verbatim from Chung et al. 2024 (doi:10.3389/fneur.2024.1389731), so that
# this ablation is comparable to their published one channel-for-channel. Not
# reordered or "tidied": a different montage would measure a different thing.
CHUNG_18 = [
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ",
]
CHUNG_4 = ["FP1-F3", "FP2-F4", "P7-O1", "P8-O2"]


def channels_for_arm(arm: int, subject_id: str) -> list[str]:
    """Which channel names this arm reads for this patient."""
    if arm == 1:
        # The patient's own clinically confirmed position, exactly as the
        # production single-channel runs use.
        return [subject_to_channel(subject_id).strip().upper()]
    if arm == 4:
        return list(CHUNG_4)
    if arm == 18:
        return list(CHUNG_18)
    raise SystemExit(f"unknown arm {arm}: expected 1, 4 or 18")


def load_fold_signals(manifest_df, raw_dir: str, edf_ids: frozenset[str],
                      channels: list[str]) -> dict[str, np.ndarray]:
    """Raw `(C, T)` per EDF, rows in the order `channels` names them.

    Row order is fixed by the requested list rather than by whatever order the
    file happens to store, because a model trained with P7-O1 on row 3 must see
    P7-O1 on row 3 in every other recording too.
    """
    from wearseizure.data.io_edf import load_edf_multichannel

    rows = manifest_df[manifest_df["edf_id"].isin(edf_ids)].drop_duplicates("edf_id")
    out: dict[str, np.ndarray] = {}
    for _, row in rows.iterrows():
        path = Path(raw_dir) / row["subject_id"] / row["edf_relpath"]
        sig, names = load_edf_multichannel(str(path), channels)
        order = {n.strip().upper(): i for i, n in enumerate(names)}
        missing = [c for c in channels if c not in order]
        if missing:
            raise SystemExit(
                f"{row['edf_id']}: channels {missing} not in this recording. The arm's "
                "montage must be present in every recording of the fold; padding the "
                "gap with zeros would train on fabricated signal."
            )
        out[row["edf_id"]] = np.ascontiguousarray(sig[[order[c] for c in channels]])
    return out


def build_arm_model(cfg: DictConfig, n_channels: int) -> torch.nn.Module:
    """The project's own model, with only `in_channels` changed.

    Everything else -- widths, kernels, dilations, the classifier -- is held
    fixed so the arms differ in exactly one thing. The parameter count does
    change, because the stem's first convolution reads more channels; that
    difference is the cost being measured and is reported per arm.
    """
    arm_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    with open_dict(arm_cfg):
        arm_cfg.model.in_channels = n_channels
    return build_model(arm_cfg)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    check_profile_data_pairing(cfg)
    ablation = cfg.get("ablation", {})
    arms = [int(a) for a in ablation.get("arms", [1, 4, 18])]
    run_tag = run_tag_from_cfg(cfg)
    seeds = seeds_from_cfg(cfg)

    manifest_df = load_manifest(str(Path(cfg.data.manifest_path)))
    records = load_records_from_manifest(manifest_df, raw_dir=str(cfg.data.raw_dir))
    folds = load_folds(str(Path(cfg.split.folds_path)),
                       expected_manifest_hash=hash_manifest(manifest_df))
    if cfg.train.get("max_folds"):
        folds = folds[: cfg.train.max_folds]

    art = Path(cfg.profile.artifacts_dir)
    raw_dir = str(cfg.data.raw_dir)
    window_s, stride_s = cfg.window.window_s, cfg.window.stride_s
    search = cfg.postprocess.get("threshold_search", {})
    grid_on = list(search.get("on_grid", [cfg.postprocess.get("threshold", 0.5)]))
    grid_off = list(search.get("off_grid", [cfg.postprocess.get("threshold", 0.5) - 0.1]))

    n_done = n_skipped = 0
    started = time.time()

    for arm in arms:
        for seed in seeds:
            out_dir = ensure_dir(art / "channel_ablation" / f"{arm}ch" / f"seed{seed}")
            for fold in folds:
                out_path = out_dir / f"{fold.fold_id}.json"
                if out_path.exists():
                    n_skipped += 1
                    continue

                subject = subject_from_fold_id(fold.fold_id, cfg.split.name)
                channels = channels_for_arm(arm, subject)
                edf_ids = fold.train_edf_ids | fold.val_edf_ids | fold.test_edf_ids

                # Refuse before spending minutes loading: a channel absent from
                # one recording of the fold makes the arm unrunnable there, and
                # discovering that after the load wastes the load.
                common = {c.strip().upper() for c in
                          fold_common_channels(manifest_df, raw_dir, edf_ids)}
                if not set(channels) <= common:
                    log.warning(
                        f"{arm}ch seed{seed} {fold.fold_id}: skipping, missing "
                        f"{sorted(set(channels) - common)}"
                    )
                    continue

                t0 = time.time()
                seed_everything(seed)
                raw = load_fold_signals(manifest_df, raw_dir, edf_ids, channels)
                signals = prepare_multichannel_signals(raw, fold.train_edf_ids)

                parts = {
                    name: MultiChannelWindowDataset(
                        signals, windows_for_partition(records, ids, window_s, stride_s))
                    for name, ids in (("train", fold.train_edf_ids),
                                      ("val", fold.val_edf_ids),
                                      ("test", fold.test_edf_ids))
                }

                model = build_arm_model(cfg, len(channels))
                n_params = sum(p.numel() for p in model.parameters())
                dl = {"num_workers": cfg.profile.get("num_workers", 0),
                      "pin_memory": str(cfg.profile.device).startswith("cuda")}
                train_loader = DataLoader(
                    parts["train"], batch_size=cfg.train.batch_size,
                    sampler=make_class_balanced_sampler(parts["train"]), **dl)
                val_loader = DataLoader(parts["val"], batch_size=cfg.train.batch_size,
                                        shuffle=False, **dl)

                trained = train_classifier(
                    model, train_loader, val_loader,
                    epochs=cfg.train.epochs, lr=cfg.train.lr,
                    weight_decay=cfg.train.weight_decay, device=cfg.profile.device,
                    early_stopping_patience=cfg.train.early_stopping_patience,
                )

                result = evaluate_fold(
                    model=trained.model, records=records, fold=fold,
                    window_s=window_s, stride_s=stride_s,
                    postprocess_method=cfg.postprocess.method,
                    postprocess_ema_alpha=cfg.postprocess.get("ema_alpha", 0.125),
                    postprocess_run_length=cfg.postprocess.get("run_length", 1),
                    postprocess_event_merge_gap_s=cfg.postprocess.get("event_merge_gap_s", 0.0),
                    threshold_on_grid=grid_on, threshold_off_grid=grid_off,
                    batch_size=cfg.train.batch_size, device=cfg.profile.device,
                    num_workers=0,
                    far_cap_per_hour=cfg.postprocess.get("far_cap_per_hour"),
                    objective=cfg.postprocess.get("objective", "max_sensitivity"),
                    postprocess_alarm_timestamp=cfg.postprocess.get(
                        "alarm_timestamp", "window_end"),
                    # The datasets are already built and already multi-channel;
                    # letting evaluate_fold rebuild them would rebuild them
                    # single-channel and score a different model than was trained.
                    datasets=parts,
                )

                payload = {
                    "arm_channels": arm, "channels": channels,
                    "fold_id": fold.fold_id, "seed": seed, "subject": subject,
                    "n_params": n_params,
                    "sensitivity": result.test_event_metrics.sensitivity,
                    "far_per_hour": result.test_event_metrics.far_per_hour,
                    # Segment-level too, because that is the level Chung et al.
                    # report the channel penalty at, and the level the question
                    # "does accuracy drop with fewer channels" is asked about.
                    "segment": {
                        "sensitivity": result.test_segment_metrics.sensitivity,
                        "specificity": result.test_segment_metrics.specificity,
                        "accuracy": result.test_segment_metrics.accuracy,
                        "balanced_accuracy": result.test_segment_metrics.balanced_accuracy,
                        "auroc": result.test_segment_metrics.auroc,
                        "prevalence": result.test_segment_metrics.prevalence,
                    },
                    "threshold_on": result.frozen_postprocess.params.threshold_on,
                    "threshold_off": result.frozen_postprocess.params.threshold_off,
                    "seconds": round(time.time() - t0, 1),
                }
                out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                n_done += 1
                log.info(
                    f"{arm:2d}ch seed{seed} {fold.fold_id}: sens="
                    f"{payload['sensitivity']:.3f} FAR={payload['far_per_hour']:.3f} "
                    f"({payload['seconds']:.0f}s, {n_params} params)"
                )

    log.info(f"{n_done} folds computed, {n_skipped} already present, "
             f"{(time.time() - started) / 3600:.2f} h")
    print("\nSummarise with:")
    print(f"  python scripts/summarise_channel_ablation.py {art}")


if __name__ == "__main__":
    main()

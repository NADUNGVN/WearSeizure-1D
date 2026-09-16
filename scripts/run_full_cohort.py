"""All 24 CHB-MIT cases, one channel, under this project's protocol.

    python scripts/run_full_cohort.py profile=server data=chbmit \
        model=wearseizure1d_k5only 'train.seeds=[0]' \
        +shard.index=0 +shard.count=4

The evaluation cohort is 13 cases because Chung et al. 2024 screened 23 and
excluded ten on clinical grounds: chb12's files lack longitudinal bipolar
montages, neurologists could not identify a seizure location in chb12, 14, 16,
18, 20 and 21, and in chb06, 09, 13 and 19 the seizure location is not near any
of the four wearable positions.

That is a defensible restriction -- it is the population a single-electrode
wearable has an indication for -- but it is also unmeasured. This script
measures it: the same recipe, the same protocol, run over every case.

**What the result will and will not mean.** For chb06, 09, 13 and 19 the
seizure is documented as not lying under any wearable electrode. A poor result
there is anatomy, not modelling, and the summary separates those cases from the
rest so the distinction survives into the report. This is not "removing a
selection bias"; it is quantifying what happens outside the device's
indication.

Which channel for the eleven unconfirmed cases
----------------------------------------------
They have no clinically confirmed position, and every recording carries all
four. Picking the best-scoring position on the evaluation data would be a leak
of exactly the kind this project exists to document, so the choice is made
**per fold, on that fold's TRAIN partition only**.

Per fold rather than once per patient, because under leave-one-seizure-out
every recording is in some fold's test set -- there is no recording that is
safe to choose on for all folds. Per-fold selection is also what a deployed
device would do: fit the electrode position during that patient's own
calibration period.

The criterion is the AUROC of per-window log-variance between ictal and
interictal windows. Seizures raise amplitude in the 1-30 Hz band, so this ranks
positions by how visible the seizure is, costs no training, and is
deterministic.

Results land in <artifacts>/full_cohort/seed<N>/<fold_id>.json, one per fold,
and a fold whose file exists is skipped -- so shards never collide and an
interrupted run resumes.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from wearseizure.data.manifest import (
    CHBMIT_WEARABLE_CHANNELS,
    is_evaluation_case,
    load_manifest,
    subject_to_channel,
)
from wearseizure.data.sampler import make_class_balanced_sampler
from wearseizure.data.splits import make_patient_specific_loso_edf
from wearseizure.models.factory import build_model
from wearseizure.training.distill import (
    MultiChannelWindowDataset,
    prepare_multichannel_signals,
    windows_for_partition,
)
from wearseizure.training.engine_baseline import evaluate_fold
from wearseizure.training.loop import train_classifier
from wearseizure.utils.env import bootstrap_env
from wearseizure.utils.logging import get_logger
from wearseizure.utils.paths import ensure_dir, seeds_from_cfg
from wearseizure.utils.profile_guard import check_profile_data_pairing
from wearseizure.utils.seeding import seed_everything

log = get_logger(__name__)
bootstrap_env(sys.argv)

# Chung et al.'s two exclusion reasons, kept apart because they mean different
# things. A wearable cannot reach the first group's seizures at all; the second
# group's seizure location is simply unknown, so a wearable might or might not
# see them.
SEIZURE_NOT_NEAR_WEARABLE = ("chb06", "chb09", "chb13", "chb19")
LOCATION_UNIDENTIFIED = ("chb12", "chb14", "chb16", "chb18", "chb20", "chb21")


def base_edf_id(edf_id: str) -> str:
    """`chb06_01@Fp1-F3` -> `chb06_01`. Fold membership is channel-independent."""
    return edf_id.split("@", 1)[0]


def combined_metadata(eval_df: pd.DataFrame, pretrain_df: pd.DataFrame) -> pd.DataFrame:
    """One row per EDF across all 24 cases, for building folds.

    The channel column is only a placeholder for the eleven unconfirmed cases --
    fold membership depends on subject, recording and seizure times, none of
    which vary with the electrode position. The actual channel is chosen later,
    per fold, on training data.
    """
    pre = pretrain_df.copy()
    pre["edf_id"] = pre["edf_id"].map(base_edf_id)
    pre = pre.drop_duplicates("edf_id")
    both = pd.concat([eval_df, pre], ignore_index=True).drop_duplicates("edf_id")
    return both.sort_values(["subject_id", "edf_id"]).reset_index(drop=True)


def auroc_logvar(signal_by_edf: dict[str, np.ndarray], windows) -> float:
    """AUROC of per-window log-variance, ictal against interictal.

    Seizures raise amplitude in the passband, so log-variance is a cheap and
    standard ictal indicator. AUROC rather than a difference of means because it
    is scale-free: the four positions sit on different amplitude scales, and a
    difference of means would rank them by amplitude rather than by separation.

    Computed by ranks rather than by a library call so the only dependency is
    numpy and the tie handling is visible.
    """
    usable = [w for w in windows if w.edf_id in signal_by_edf]
    if not usable:
        return float("nan")
    labels = np.array([w.label for w in usable])
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    vals = np.array([
        np.log(np.var(signal_by_edf[w.edf_id][w.start_idx:w.end_idx]) + 1e-12)
        for w in usable
    ])
    order = np.argsort(vals, kind="stable")
    ranks = np.empty(len(vals), dtype=np.float64)
    ranks[order] = np.arange(1, len(vals) + 1)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def select_channel(signals: dict[str, np.ndarray], channels: list[str],
                   records, train_edf_ids, window_s: float,
                   stride_s: float) -> tuple[str, dict]:
    """Pick the wearable position whose seizures separate best on TRAIN data.

    Never on validation or test: choosing an electrode on data the model is
    scored against is the same class of leak this project exists to document.
    """
    train_windows = windows_for_partition(records, train_edf_ids, window_s, stride_s)
    if not train_windows:
        raise SystemExit("no training windows: cannot choose a channel")

    scores = {
        name: auroc_logvar({e: signals[e][row] for e in signals}, train_windows)
        for row, name in enumerate(channels)
    }
    usable = {k: v for k, v in scores.items() if not np.isnan(v)}
    if not usable:
        raise SystemExit(
            f"every candidate position scored NaN for this fold ({scores}); its "
            "training partition has no seizure windows, or no recording carries "
            "the montage."
        )
    # Sorted first so ties break on channel name rather than on dict order --
    # the choice has to be reproducible across runs and machines.
    best = max(sorted(usable), key=lambda k: usable[k])
    return best, scores


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    check_profile_data_pairing(cfg)
    shard = cfg.get("shard", {})
    shard_i, shard_n = int(shard.get("index", 0)), int(shard.get("count", 1))
    if not 0 <= shard_i < shard_n:
        raise SystemExit(f"shard.index={shard_i} outside 0..{shard_n - 1}")

    seeds = seeds_from_cfg(cfg)
    art = Path(cfg.profile.artifacts_dir)
    raw_dir = str(cfg.data.raw_dir)
    window_s, stride_s = cfg.window.window_s, cfg.window.stride_s
    channels = [c.strip().upper() for c in CHBMIT_WEARABLE_CHANNELS]

    eval_df = load_manifest(str(Path(cfg.data.manifest_path)))
    pre_path = Path(cfg.data.get("pretrain_manifest_path", ""))
    if not pre_path.is_file():
        raise SystemExit(
            f"no pre-training manifest at {pre_path}. It carries the eleven "
            "non-Appendix-A cases; build it with scripts/make_manifest.py first."
        )
    pre_df = load_manifest(str(pre_path))
    meta = combined_metadata(eval_df, pre_df)
    log.info(f"{meta['subject_id'].nunique()} cases, {len(meta)} recordings")

    folds = make_patient_specific_loso_edf(meta, seed=0)
    folds = [f for i, f in enumerate(folds) if i % shard_n == shard_i]
    log.info(f"shard {shard_i}/{shard_n}: {len(folds)} folds")

    search = cfg.postprocess.get("threshold_search", {})
    grid_on = list(search.get("on_grid", [cfg.postprocess.get("threshold", 0.5)]))
    grid_off = list(search.get("off_grid", [cfg.postprocess.get("threshold", 0.5) - 0.1]))

    from wearseizure.data.io_edf import load_edf_multichannel
    from wearseizure.data.loader import load_records_from_manifest

    n_done = n_skipped = 0
    for seed in seeds:
        out_dir = ensure_dir(art / "full_cohort" / f"seed{seed}")
        for fold in folds:
            out_path = out_dir / f"{fold.fold_id}.json"
            if out_path.exists():
                n_skipped += 1
                continue

            subject = fold.fold_id.split("__")[0]
            confirmed = is_evaluation_case(subject)
            edf_ids = fold.train_edf_ids | fold.val_edf_ids | fold.test_edf_ids
            rows = meta[meta["edf_id"].isin(edf_ids)]
            t0 = time.time()
            seed_everything(seed)

            # Records carry metadata only here -- seizure times, duration,
            # sampling rate. The signal comes from the multi-channel load
            # below, because the channel is not known until it is chosen.
            records = load_records_from_manifest(rows, raw_dir=raw_dir)

            raw = {}
            for _, row in rows.iterrows():
                path = Path(raw_dir) / row["subject_id"] / row["edf_relpath"]
                sig, names = load_edf_multichannel(str(path), channels)
                order = {n.strip().upper(): i for i, n in enumerate(names)}
                if not set(channels) <= set(order):
                    log.warning(f"{row['edf_id']}: missing "
                                f"{sorted(set(channels) - set(order))}, skipping fold")
                    raw = {}
                    break
                raw[row["edf_id"]] = np.ascontiguousarray(sig[[order[c] for c in channels]])
            if not raw:
                continue

            if confirmed:
                chosen = subject_to_channel(subject).strip().upper()
                scores = {"confirmed": chosen}
            else:
                chosen, scores = select_channel(
                    raw, channels, records, fold.train_edf_ids, window_s, stride_s)
                log.info(f"{fold.fold_id}: chose {chosen} from "
                         + ", ".join(f"{k}={v:.3f}" for k, v in sorted(scores.items())))

            row_idx = channels.index(chosen)
            one = {e: raw[e][row_idx:row_idx + 1] for e in raw}
            signals = prepare_multichannel_signals(one, fold.train_edf_ids)
            parts = {
                name: MultiChannelWindowDataset(
                    signals, windows_for_partition(records, ids, window_s, stride_s))
                for name, ids in (("train", fold.train_edf_ids),
                                  ("val", fold.val_edf_ids),
                                  ("test", fold.test_edf_ids))
            }

            model = build_model(cfg)
            dl = {"num_workers": cfg.profile.get("num_workers", 0),
                  "pin_memory": str(cfg.profile.device).startswith("cuda")}
            trained = train_classifier(
                model,
                DataLoader(parts["train"], batch_size=cfg.train.batch_size,
                           sampler=make_class_balanced_sampler(parts["train"]), **dl),
                DataLoader(parts["val"], batch_size=cfg.train.batch_size,
                           shuffle=False, **dl),
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
                batch_size=cfg.train.batch_size, device=cfg.profile.device, num_workers=0,
                far_cap_per_hour=cfg.postprocess.get("far_cap_per_hour"),
                objective=cfg.postprocess.get("objective", "max_sensitivity"),
                postprocess_alarm_timestamp=cfg.postprocess.get("alarm_timestamp", "window_end"),
                datasets=parts,
            )

            payload = {
                "fold_id": fold.fold_id, "seed": seed, "subject": subject,
                "clinically_confirmed": bool(confirmed),
                "exclusion_group": (
                    "seizure_not_near_wearable" if subject in SEIZURE_NOT_NEAR_WEARABLE
                    else "location_unidentified" if subject in LOCATION_UNIDENTIFIED
                    else "confirmed" if confirmed else "other_excluded"),
                "channel": chosen, "channel_scores": scores,
                "sensitivity": result.test_event_metrics.sensitivity,
                "far_per_hour": result.test_event_metrics.far_per_hour,
                "segment": {
                    "sensitivity": result.test_segment_metrics.sensitivity,
                    "accuracy": result.test_segment_metrics.accuracy,
                    "auroc": result.test_segment_metrics.auroc,
                },
                "seconds": round(time.time() - t0, 1),
            }
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            n_done += 1
            log.info(f"{fold.fold_id} [{chosen}]: sens={payload['sensitivity']:.3f} "
                     f"FAR={payload['far_per_hour']:.3f} ({payload['seconds']:.0f}s)")

    log.info(f"{n_done} folds computed, {n_skipped} already present")
    print("\nSummarise with:")
    print(f"  python scripts/summarise_full_cohort.py {art}")


if __name__ == "__main__":
    main()

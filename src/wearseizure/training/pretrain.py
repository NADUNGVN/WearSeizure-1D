"""Cohort pre-training, then per-patient fine-tuning (lever L1 of
`docs/RESEARCH_REALITY_CHECK.md` section 14).

The problem this fixes
----------------------
`data/splits.make_patient_specific_loso_edf` builds every fold inside
`manifest_df.groupby("subject_id")`, so a fold's `train_edf_ids` contain **only
the target patient's own EDFs**. `scripts/train.py` then calls `build_model()`
per fold, i.e. from random initialisation. Put together: each of the 66 folds
trains a ~12k-parameter CNN from scratch on one patient's handful of
recordings. For chb17 (3 seizures total, one held out per fold) that is a model
learning from **two seizures**.

That single fact explains the pattern in `docs/EXPERIMENT_LOG_G1a.md`: the
worst-patient gate always lands on the patients with the fewest seizures
(chb17/chb02/chb04, 3-4 events each); more epochs helped unevenly because the
models overfit; `val_loss` was too noisy to early-stop on; and 20 runs of
threshold/postprocess/architecture search plateaued, because the classifier was
starved of data rather than badly calibrated.

What this module does
---------------------
For a fold belonging to patient S, pre-train one model on **every other
patient** and use it as the initialisation for S's folds, which then fine-tune
on S's own remaining EDFs exactly as before.

Leakage safety: the pre-training corpus is built from subjects != S, so no
sample of S -- and in particular no sample of the held-out test EDF -- is ever
seen during pre-training. The fine-tuning stage is byte-identical to the
existing path. `cohort_pretrain_fold` asserts the exclusion rather than trusting
it, and `tests/unit/test_pretrain_cohort.py` pins it.

Cost note: the pre-trained initialisation depends only on *which patient is
held out*, not on which of that patient's EDFs is the fold's test file. So 66
folds need only 13 pre-trainings, cached on disk and reused.

Normalisation: pre-training fits its own affine normaliser on the cohort (via
`build_fold_datasets`), while fine-tuning keeps fitting on the fold's own train
partition. The two stages therefore see slightly different input scaling, which
is deliberate -- it mirrors deployment, where a device is calibrated per wearer,
and it leaves the existing per-fold path untouched.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from wearseizure.data.dataset import build_fold_datasets
from wearseizure.data.manifest import hash_manifest
from wearseizure.data.records import EEGRecord
from wearseizure.data.sampler import make_class_balanced_sampler
from wearseizure.data.splits import Fold, validate_fold
from wearseizure.training.loop import train_classifier
from wearseizure.utils.logging import get_logger
from wearseizure.utils.seeding import rng_for

log = get_logger(__name__)


def cohort_pretrain_fold(
    manifest_df: pd.DataFrame,
    held_out_subject: str,
    seed: int,
    val_subject_fraction: float = 0.2,
) -> Fold:
    """A fold whose train/val partitions cover every subject EXCEPT
    `held_out_subject`, and whose test partition is empty.

    The split is at the **subject** level, so early stopping during
    pre-training is measured on patients the pre-trained model has never seen —
    the same discipline the zero-shot split uses.

    Test is left empty on purpose: `build_fold_datasets` filters and normalises
    every EDF it is given, and the held-out subject's signals are not needed
    here. Including them would copy ~1/13 of the corpus 13 times for nothing.
    """
    manifest_hash = hash_manifest(manifest_df)
    subjects = sorted(manifest_df["subject_id"].unique())
    if held_out_subject not in subjects:
        raise ValueError(f"subject {held_out_subject!r} not in manifest (have {subjects})")

    remaining = [s for s in subjects if s != held_out_subject]
    if len(remaining) < 2:
        raise ValueError(
            f"cohort pre-training for {held_out_subject!r} needs >=2 other subjects "
            f"(1 train + 1 val), found {len(remaining)}"
        )

    rng = rng_for(held_out_subject, "cohort_pretrain_val", base_seed=seed)
    n_val = min(max(1, round(len(remaining) * val_subject_fraction)), len(remaining) - 1)
    val_idx = rng.choice(len(remaining), size=n_val, replace=False)
    val_subjects = {remaining[i] for i in val_idx}
    train_subjects = [s for s in remaining if s not in val_subjects]

    val_ids = frozenset(manifest_df.loc[manifest_df["subject_id"].isin(val_subjects), "edf_id"])
    train_ids = frozenset(manifest_df.loc[manifest_df["subject_id"].isin(train_subjects), "edf_id"])

    fold = Fold(
        fold_id=f"pretrain__{held_out_subject}",
        train_edf_ids=train_ids,
        val_edf_ids=val_ids,
        test_edf_ids=frozenset(),
        held_out_key=held_out_subject,
        manifest_hash=manifest_hash,
    )
    validate_fold(fold)

    # Assert the exclusion instead of trusting the filter above: this is the
    # one property that makes cohort pre-training leakage-safe at all.
    held_out_ids = frozenset(
        manifest_df.loc[manifest_df["subject_id"] == held_out_subject, "edf_id"]
    )
    contaminated = held_out_ids & (fold.train_edf_ids | fold.val_edf_ids)
    if contaminated:
        raise ValueError(
            f"cohort pre-training for {held_out_subject!r} would train on that subject's own "
            f"EDF(s) {sorted(contaminated)} -- refusing"
        )
    return fold


def _cache_paths(cache_dir: Path, held_out_subject: str) -> tuple[Path, Path]:
    return cache_dir / f"{held_out_subject}.pt", cache_dir / f"{held_out_subject}.json"


def get_or_train_cohort_init(
    records: dict[str, EEGRecord],
    manifest_df: pd.DataFrame,
    held_out_subject: str,
    model_factory,
    cache_dir: Path,
    window_s: float,
    stride_s: float,
    seed: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    device: str,
    early_stopping_patience: int,
    num_workers: int = 0,
    val_subject_fraction: float = 0.2,
    class_balanced_sampling: bool = True,
    prefetch_factor: int = 4,
    force: bool = False,
) -> dict:
    """Return a cohort-pre-trained `state_dict` for `held_out_subject`,
    training and caching it on first use.

    `model_factory` is a zero-argument callable so the cached weights always
    match the model config in play; a cache hit for a different architecture is
    impossible because `cache_dir` is keyed by model and window name.
    """
    ckpt_path, meta_path = _cache_paths(cache_dir, held_out_subject)
    if ckpt_path.exists() and not force:
        log.info(f"cohort pre-train: reusing cached init for {held_out_subject} <- {ckpt_path}")
        return torch.load(ckpt_path, map_location="cpu", weights_only=True)

    fold = cohort_pretrain_fold(manifest_df, held_out_subject, seed, val_subject_fraction)
    datasets, _band, _normalizer = build_fold_datasets(records, fold, window_s, stride_s)
    train_ds, val_ds = datasets["train"], datasets["val"]
    log.info(
        f"cohort pre-train for {held_out_subject}: "
        f"{len(fold.train_edf_ids)} train EDFs / {len(fold.val_edf_ids)} val EDFs, "
        f"{len(train_ds)} train windows / {len(val_ds)} val windows"
    )

    dl_kwargs = {
        "num_workers": num_workers,
        "pin_memory": device.startswith("cuda"),
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        # Deeper queue than the default 2. A pre-training epoch is ~2M windows
        # for a model whose per-iteration GPU time is dominated by kernel-launch
        # overhead, so the loader must never be the thing the GPU waits on.
        # Cost is bounded and small: prefetch_factor * num_workers * batch_size
        # windows in flight, at 4KB per window (1024 samples, float32).
        dl_kwargs["prefetch_factor"] = prefetch_factor
    if class_balanced_sampling:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=make_class_balanced_sampler(train_ds), **dl_kwargs
        )
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **dl_kwargs)

    model = model_factory()
    result = train_classifier(
        model, train_loader, val_loader, epochs=epochs, lr=lr, weight_decay=weight_decay,
        device=device, early_stopping_patience=early_stopping_patience,
    )

    state = {k: v.detach().cpu() for k, v in result.model.state_dict().items()}
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, ckpt_path)
    meta_path.write_text(
        json.dumps(
            {
                "held_out_subject": held_out_subject,
                "manifest_hash": fold.manifest_hash,
                "n_train_edfs": len(fold.train_edf_ids),
                "n_val_edfs": len(fold.val_edf_ids),
                "n_train_windows": len(train_ds),
                "n_val_windows": len(val_ds),
                "best_val_loss": result.best_val_loss,
                "epochs_run": len(result.history),
                "epochs_max": epochs,
                "lr": lr,
                "seed": seed,
                "window_s": window_s,
                "stride_s": stride_s,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    log.info(
        f"cohort pre-train for {held_out_subject}: best_val_loss={result.best_val_loss:.4f} "
        f"after {len(result.history)} epochs -> {ckpt_path}"
    )
    return state

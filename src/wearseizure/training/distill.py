"""Lever L3: distil a multi-channel teacher into the single-channel student.

Why this and not more data
--------------------------
Lever L5 established that the pre-training corpus cannot usefully be extended
past the 13 clinically-confirmed CHB-MIT cases: widening it changed nothing at
four electrode positions and made things significantly worse at one
(`docs/EXPERIMENT_LOG_G1a.md` section 2e). So the remaining room is not in
*more* recordings but in *more of each recording*. A teacher reading all
channels of the same EDFs solves a materially easier problem, and its soft
probabilities carry what the hard label does not -- how confident it is, and on
which windows -- which is exactly what one channel is short of.

Why the logits are precomputed
------------------------------
The teacher could run alongside the student, but then every student epoch pays
for an 18-channel forward pass, the two models must be co-resident in VRAM, and
the teacher is re-evaluated identically for every seed. Instead the teacher is
trained once per fold, run once over that fold's train windows, and its logits
are stored as an `(N, 2)` float32 array. That is about 1.3 MB for a typical
fold, it is reused across student seeds, and the student's training loop stays
a single-channel loop with one extra tensor per batch.

Alignment is exact rather than approximate: `data/windowing.Window` is built
from `record.meta` (sampling rate, seizure times) and index arithmetic alone,
never from signal values, so the 18-channel view of an EDF produces exactly the
same window list as the 1-channel view.

Leakage safety
--------------
The teacher is trained on the fold's own train partition and nothing else, with
the same causal filter and the same train-fitted normaliser discipline as the
student. It never sees val or test, and it never appears at evaluation time --
`build_fold_datasets` attaches logits to the train partition only. The reported
system is a single-channel detector; the teacher is a training-time scaffold.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from wearseizure.data.records import EEGRecord
from wearseizure.data.splits import Fold
from wearseizure.data.windowing import Window, windows_for_edf
from wearseizure.models.teacher import build_teacher
from wearseizure.signal.filters import CausalBandpass
from wearseizure.signal.normalize import fit_affine_normalizer
from wearseizure.utils.logging import get_logger

log = get_logger(__name__)


class MultiChannelWindowDataset(Dataset):
    """The same windows as `WearSeizureWindowDataset`, but `(C, L)` per item.

    Takes the window list from the caller rather than recomputing it, so the
    teacher and the student are guaranteed to be looking at the same windows in
    the same order -- the property the precomputed logit array depends on.
    """

    def __init__(self, signals: dict[str, np.ndarray], windows: list[Window]) -> None:
        self.signals = signals
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        w = self.windows[idx]
        x = self.signals[w.edf_id][:, w.start_idx : w.end_idx]
        return torch.from_numpy(np.ascontiguousarray(x)), torch.tensor(w.label, dtype=torch.long)


def prepare_multichannel_signals(
    raw: dict[str, np.ndarray],
    train_edf_ids: frozenset[str],
    band_low_hz: float = 1.0,
    band_high_hz: float = 30.0,
) -> dict[str, np.ndarray]:
    """Filter and normalise `(C, T)` signals with the same discipline the
    single-channel path uses: one causal pass per channel with state reset at
    t=0, and one affine normaliser fitted on the TRAIN partition only.

    A single normaliser is fitted across all channels rather than one per
    channel, because the channels share a physical scale (microvolts from the
    same amplifier) and per-channel scaling would hand the teacher a
    normalisation the student's single channel could never reproduce.
    """
    band = CausalBandpass(low_hz=band_low_hz, high_hz=band_high_hz)
    filtered = {
        edf_id: np.stack([band.apply_full_reset(ch) for ch in sig], axis=0)
        for edf_id, sig in raw.items()
    }
    missing = train_edf_ids - filtered.keys()
    if missing:
        raise ValueError(f"multi-channel signals missing for train EDFs {sorted(missing)}")
    normalizer = fit_affine_normalizer(
        [filtered[edf_id].ravel() for edf_id in sorted(train_edf_ids)]
    )
    out: dict[str, np.ndarray] = {}
    for edf_id in sorted(filtered):
        sig = filtered.pop(edf_id)
        out[edf_id] = normalizer.apply(sig).astype(np.float32, copy=False)
    return out


def windows_for_partition(
    records: dict[str, EEGRecord], edf_ids: frozenset[str], window_s: float, stride_s: float
) -> list[Window]:
    """Window list for a partition, in the same order `WearSeizureWindowDataset`
    builds it -- sorted by `edf_id`, then chronological within each EDF."""
    windows: list[Window] = []
    for edf_id in sorted(edf_ids):
        windows.extend(windows_for_edf(records[edf_id], window_s, stride_s, edf_ids))
    return windows


def teacher_logits_for_windows(
    teacher: torch.nn.Module,
    signals: dict[str, np.ndarray],
    windows: list[Window],
    device: str = "cpu",
    batch_size: int = 256,
    num_workers: int = 0,
) -> np.ndarray:
    """`(N, 2)` float32 logits, one row per window, in the given window order."""
    teacher.eval().to(device)
    loader = DataLoader(
        MultiChannelWindowDataset(signals, windows),
        batch_size=batch_size,
        shuffle=False,          # order IS the alignment; never shuffle here
        num_workers=num_workers,
    )
    out = np.empty((len(windows), 2), dtype=np.float32)
    i = 0
    with torch.no_grad():
        for x, _ in loader:
            logits = teacher(x.to(device, non_blocking=True))
            n = logits.shape[0]
            out[i : i + n] = logits.detach().cpu().numpy()
            i += n
    if i != len(windows):
        raise RuntimeError(f"scored {i} windows but expected {len(windows)}")
    return out


def fold_teacher_logits(
    records: dict[str, EEGRecord],
    multichannel_raw: dict[str, np.ndarray],
    fold: Fold,
    teacher_factory,
    window_s: float,
    stride_s: float,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    device: str,
    early_stopping_patience: int,
    num_workers: int = 0,
    class_balanced_sampling: bool = True,
) -> np.ndarray:
    """Train a teacher on this fold's train partition and return its logits over
    that partition's windows.

    The val partition is used for the teacher's own early stopping, which is the
    same partition the student later fits thresholds on. That is deliberate and
    it is not leakage into the test set -- but it does mean the teacher has seen
    val, so the teacher's own numbers must never be reported as a result.
    """
    from wearseizure.data.sampler import make_class_balanced_sampler
    from wearseizure.training.loop import train_classifier

    signals = prepare_multichannel_signals(multichannel_raw, fold.train_edf_ids)
    train_windows = windows_for_partition(records, fold.train_edf_ids, window_s, stride_s)
    val_windows = windows_for_partition(records, fold.val_edf_ids, window_s, stride_s)

    n_channels = next(iter(signals.values())).shape[0]
    log.info(
        f"teacher for fold {fold.fold_id}: {n_channels} channels, "
        f"{len(train_windows)} train / {len(val_windows)} val windows"
    )

    train_ds = MultiChannelWindowDataset(signals, train_windows)
    val_ds = MultiChannelWindowDataset(signals, val_windows)
    dl = {"num_workers": num_workers, "pin_memory": device.startswith("cuda")}
    if class_balanced_sampling:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=make_class_balanced_sampler(train_ds), **dl
        )
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **dl)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **dl)

    result = train_classifier(
        teacher_factory(n_channels), train_loader, val_loader,
        epochs=epochs, lr=lr, weight_decay=weight_decay, device=device,
        early_stopping_patience=early_stopping_patience,
    )
    log.info(f"teacher for fold {fold.fold_id}: best_val_loss={result.best_val_loss:.4f}")
    return teacher_logits_for_windows(
        result.model, signals, train_windows, device=device, batch_size=batch_size
    )


def load_multichannel_for_fold(
    manifest_df, raw_dir: str, edf_ids: frozenset[str]
) -> dict[str, np.ndarray]:
    """Every channel of each EDF in `edf_ids`, as `(C, T)` float32.

    Uses the manifest's own `subject_id` / `edf_relpath` rather than a second
    manifest: the evaluation manifest already records where each file is, and
    the only thing that differs for the teacher is how many channels come back
    out of it. A separate multi-channel manifest would be one more thing that
    can drift out of step with the one the splits are version-locked to.
    """
    from wearseizure.data.io_edf import load_edf_multichannel

    rows = manifest_df[manifest_df["edf_id"].isin(edf_ids)]
    signals: dict[str, np.ndarray] = {}
    channel_counts: dict[int, int] = {}
    for _, row in rows.iterrows():
        path = Path(raw_dir) / row["subject_id"] / row["edf_relpath"]
        sig, names = load_edf_multichannel(str(path))
        signals[row["edf_id"]] = sig
        channel_counts[len(names)] = channel_counts.get(len(names), 0) + 1

    missing = set(edf_ids) - signals.keys()
    if missing:
        raise ValueError(f"multi-channel load found no rows for {sorted(missing)}")
    if len(channel_counts) > 1:
        # CHB-MIT is not uniform: a few files carry a different montage. Padding
        # or truncating to a common width would feed the teacher fabricated or
        # silently reordered channels, so this is refused rather than patched.
        raise ValueError(
            f"EDFs in this fold have differing channel counts {channel_counts}; "
            "the teacher needs one consistent montage per fold"
        )
    return signals


def _teacher_cache_paths(cache_dir: Path, fold_id: str) -> tuple[Path, Path]:
    return cache_dir / f"{fold_id}.npz", cache_dir / f"{fold_id}.json"


def get_or_train_fold_teacher_logits(
    records: dict[str, EEGRecord],
    manifest_df,
    raw_dir: str,
    fold: Fold,
    cache_dir: Path,
    window_s: float,
    stride_s: float,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    device: str,
    early_stopping_patience: int,
    num_workers: int = 0,
    class_balanced_sampling: bool = True,
    force: bool = False,
) -> np.ndarray:
    """Teacher logits for a fold, training and caching them on first use.

    Cached by FOLD, not by seed. The teacher depends on the fold's train
    partition, the window geometry and its own hyperparameters -- none of which
    the student's seed touches -- so training one per seed would be three times
    the cost for three answers to the same question. It also makes the soft
    targets identical across seeds, which is what lets a seed-to-seed comparison
    isolate the student.

    The cache carries the settings it was built from and is rebuilt when they
    change, the same guard `training/pretrain.py` uses: reusing logits from a
    different teacher configuration would look like "distillation did nothing"
    while the distillation being measured never ran.
    """
    ckpt, meta_path = _teacher_cache_paths(cache_dir, fold.fold_id)
    signature = {
        "fold_id": fold.fold_id,
        "manifest_hash": fold.manifest_hash,
        "window_s": window_s,
        "stride_s": stride_s,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "early_stopping_patience": early_stopping_patience,
        "class_balanced_sampling": class_balanced_sampling,
    }
    if ckpt.exists() and not force:
        cached = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        if cached.get("signature") == signature:
            log.info(f"teacher: reusing cached logits for {fold.fold_id} <- {ckpt}")
            return np.load(ckpt)["logits"]
        log.warning(
            f"teacher: cached logits for {fold.fold_id} were built from different settings; "
            "re-training"
        )

    signals = load_multichannel_for_fold(manifest_df, raw_dir, fold.train_edf_ids | fold.val_edf_ids)
    logits = fold_teacher_logits(
        records=records, multichannel_raw=signals, fold=fold,
        teacher_factory=build_teacher, window_s=window_s, stride_s=stride_s,
        epochs=epochs, lr=lr, weight_decay=weight_decay, batch_size=batch_size,
        device=device, early_stopping_patience=early_stopping_patience,
        num_workers=num_workers, class_balanced_sampling=class_balanced_sampling,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ckpt, logits=logits)
    meta_path.write_text(
        json.dumps(
            {"signature": signature, "n_windows": len(logits),
             "n_channels": int(next(iter(signals.values())).shape[0])},
            indent=2, sort_keys=True,
        ),
        encoding="utf-8",
    )
    log.info(f"teacher: {len(logits)} logits for {fold.fold_id} -> {ckpt}")
    return logits

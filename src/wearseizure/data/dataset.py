"""Torch Dataset assembled per-fold, per-partition.

Filtering (causal, per-EDF state reset) and normalization (affine, fit on
train only) happen once at construction time; windowing happens last, and
only over the edf_ids belonging to the requested partition of the given fold
-- so a bug that mixed up partitions would surface immediately as a
`ValueError` from `windows_for_edf`, not as silent leakage.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Literal

import torch
from torch.utils.data import Dataset

from wearseizure.data.records import EEGRecord
from wearseizure.data.splits import Fold
from wearseizure.data.windowing import Window, extract, windows_for_edf
from wearseizure.signal.filters import CausalBandpass
from wearseizure.signal.normalize import AffineNormalizer, fit_affine_normalizer

Partition = Literal["train", "val", "test"]


def _partition_edf_ids(fold: Fold, partition: Partition) -> frozenset[str]:
    return {"train": fold.train_edf_ids, "val": fold.val_edf_ids, "test": fold.test_edf_ids}[partition]


def _filter_records(
    records: dict[str, EEGRecord], edf_ids, band: CausalBandpass
) -> dict[str, EEGRecord]:
    return {edf_id: replace(records[edf_id], signal=band.apply_full_reset(records[edf_id].signal)) for edf_id in edf_ids}


class WearSeizureWindowDataset(Dataset):
    def __init__(
        self,
        filtered_normalized_records: dict[str, EEGRecord],
        allowed_edf_ids: frozenset[str],
        window_s: float,
        stride_s: float,
    ) -> None:
        self.records = filtered_normalized_records
        self.windows: list[Window] = []
        for edf_id in sorted(allowed_edf_ids):
            record = filtered_normalized_records[edf_id]
            self.windows.extend(windows_for_edf(record, window_s, stride_s, allowed_edf_ids))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        w = self.windows[idx]
        x = extract(self.records[w.edf_id], w)
        return torch.from_numpy(x).float().unsqueeze(0), torch.tensor(w.label, dtype=torch.long)

    def window_at(self, idx: int) -> Window:
        return self.windows[idx]


def build_fold_datasets(
    records: dict[str, EEGRecord],
    fold: Fold,
    window_s: float,
    stride_s: float,
    band_low_hz: float = 1.0,
    band_high_hz: float = 30.0,
) -> tuple[dict[Partition, WearSeizureWindowDataset], CausalBandpass, AffineNormalizer]:
    """Fit filter/normalizer on `fold.train_edf_ids` only, then build all three
    partitions' datasets from that single frozen (band, normalizer) pair.
    """
    band = CausalBandpass(low_hz=band_low_hz, high_hz=band_high_hz)

    filtered_train = _filter_records(records, fold.train_edf_ids, band)
    if not filtered_train:
        raise ValueError(f"fold {fold.fold_id} has an empty train partition")
    normalizer = fit_affine_normalizer([r.signal for r in filtered_train.values()])

    all_ids = fold.train_edf_ids | fold.val_edf_ids | fold.test_edf_ids
    filtered_all = _filter_records(records, all_ids, band)
    normalized_all = {
        edf_id: replace(r, signal=normalizer.apply(r.signal)) for edf_id, r in filtered_all.items()
    }

    datasets: dict[Partition, WearSeizureWindowDataset] = {}
    for partition in ("train", "val", "test"):
        allowed = _partition_edf_ids(fold, partition)
        datasets[partition] = WearSeizureWindowDataset(normalized_all, allowed, window_s, stride_s)
    return datasets, band, normalizer

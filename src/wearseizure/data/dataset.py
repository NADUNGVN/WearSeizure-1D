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

import numpy as np
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

    def __getitems__(self, indices: list[int]):
        """Fetch a whole batch in one call (PyTorch's batched-fetch protocol).

        `DataLoader`'s map-style fetcher uses this when it exists, instead of
        calling `__getitem__` once per sample. That matters here far more than
        it usually would: the model is ~12k parameters, so a batch costs the
        GPU well under a millisecond, and measured on this data a batch of 256
        spent ~1.7ms in per-sample Python before any tensor work happened --
        the loader, not the device, was the bottleneck, which is why GPU and
        memory-bandwidth counters both sat low while 14 workers were busy.

        Filling one preallocated float32 array and stacking once collapses 256
        interpreter round-trips into a tight loop over `numpy` slices.
        """
        n = len(indices)
        first = self.windows[indices[0]] if n else None
        window_len = first.n_samples if first is not None else 0
        batch = np.empty((n, window_len), dtype=np.float32)
        labels = np.empty(n, dtype=np.int64)
        for row, idx in enumerate(indices):
            w = self.windows[idx]
            batch[row] = self.records[w.edf_id].signal[w.start_idx : w.end_idx]
            labels[row] = w.label
        x = torch.from_numpy(batch).unsqueeze(1)
        y = torch.from_numpy(labels)
        return list(zip(x, y))

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

    if not fold.train_edf_ids:
        raise ValueError(f"fold {fold.fold_id} has an empty train partition")

    # Filter ONCE. This used to filter the train partition, then filter every
    # partition again -- so the train EDFs were filtered twice and both float64
    # copies stayed alive at the same time. Harmless at 553h; at the lever-L5
    # corpus's 2085h it was one of the allocations that OOM-killed Phase 3.
    all_ids = fold.train_edf_ids | fold.val_edf_ids | fold.test_edf_ids
    filtered_all = _filter_records(records, all_ids, band)

    # Sorted, so the concatenation order the normalizer sees does not depend on
    # frozenset iteration order. Median and MAD are order-independent in value,
    # so this changes nothing numerically -- it just makes the fit reproducible.
    normalizer = fit_affine_normalizer(
        [filtered_all[edf_id].signal for edf_id in sorted(fold.train_edf_ids)]
    )

    # Store float32, not the float64 that scipy's lfilter and the normalizer
    # produce. Every window was already being cast to float32 on its way into
    # the model, so this is bit-identical -- it just happens once per EDF
    # instead of once per window, and it halves both the resident dataset
    # (~4.4GB -> ~2.2GB for the 13-case cohort) and the bytes each batch moves.
    #
    # Built by draining `filtered_all` rather than comprehending over it, so the
    # float64 originals are freed as the float32 copies appear instead of both
    # corpora being resident at once.
    normalized_all: dict[str, EEGRecord] = {}
    for edf_id in sorted(filtered_all):
        r = filtered_all.pop(edf_id)
        normalized_all[edf_id] = replace(
            r, signal=normalizer.apply(r.signal).astype(np.float32, copy=False)
        )

    datasets: dict[Partition, WearSeizureWindowDataset] = {}
    for partition in ("train", "val", "test"):
        allowed = _partition_edf_ids(fold, partition)
        datasets[partition] = WearSeizureWindowDataset(normalized_all, allowed, window_s, stride_s)
    return datasets, band, normalizer

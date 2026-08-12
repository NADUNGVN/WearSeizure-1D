"""Causal windowing -- the only place raw samples are cut into model inputs.

Memo 5.1, step 2: "Split theo EDF/seizure event truoc khi loc, normalize va
windowing." Windowing therefore *requires* the caller to already know which
`edf_id`s belong to the current fold partition (train/val/test); calling it on
an EDF outside the given allow-list is a programming error, not a warning.

Every window ends at `end_sec` (its decision point) and never reads samples
beyond it -- the causal, no-lookahead property required by memo 5.1 step 5.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from wearseizure.data.records import EEGRecord


@dataclass(frozen=True)
class Window:
    edf_id: str
    subject_id: str
    start_idx: int
    end_idx: int
    start_sec: float
    end_sec: float
    label: int  # segment-level: 1 if window overlaps any seizure event

    @property
    def n_samples(self) -> int:
        return self.end_idx - self.start_idx


def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return a_start < b_end and b_start < a_end


def windows_for_edf(
    record: EEGRecord,
    window_s: float,
    stride_s: float,
    fold_partition_edf_ids: set[str],
) -> Iterator[Window]:
    if record.meta.edf_id not in fold_partition_edf_ids:
        raise ValueError(
            f"windows_for_edf: {record.meta.edf_id!r} is not in the allowed partition "
            "for this fold. Windowing an EDF outside its assigned fold partition "
            "would leak information across train/val/test; refusing."
        )
    fs = record.meta.fs_hz
    window_len = round(window_s * fs)
    stride_len = round(stride_s * fs)
    if window_len <= 0 or stride_len <= 0:
        raise ValueError(f"window_s={window_s}, stride_s={stride_s} produced non-positive lengths")
    n = len(record.signal)
    events = [(e.onset_sec, e.offset_sec) for e in record.meta.seizure_events]

    end_idx = window_len
    while end_idx <= n:
        start_idx = end_idx - window_len
        start_sec = start_idx / fs
        end_sec = end_idx / fs
        label = int(any(_overlaps(start_sec, end_sec, on, off) for on, off in events))
        yield Window(
            edf_id=record.meta.edf_id,
            subject_id=record.meta.subject_id,
            start_idx=start_idx,
            end_idx=end_idx,
            start_sec=start_sec,
            end_sec=end_sec,
            label=label,
        )
        end_idx += stride_len


def extract(record: EEGRecord, window: Window) -> np.ndarray:
    if window.edf_id != record.meta.edf_id:
        raise ValueError(f"window.edf_id {window.edf_id!r} != record edf_id {record.meta.edf_id!r}")
    return record.signal[window.start_idx : window.end_idx]

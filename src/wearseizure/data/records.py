"""Shared in-memory record type so the rest of the pipeline cannot tell whether
a recording came from a real EDF (`io_edf.py`) or the synthetic generator
(`synthetic.py`) -- both produce the same `EEGRecord`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wearseizure.data.manifest import EEGRecordMeta


@dataclass
class EEGRecord:
    meta: EEGRecordMeta
    signal: np.ndarray  # raw, unfiltered, shape (n_samples,), units matching meta.fs_hz

    def __post_init__(self) -> None:
        if self.signal.ndim != 1:
            raise ValueError(f"{self.meta.edf_id}: signal must be 1-D, got shape {self.signal.shape}")
        if len(self.signal) != self.meta.n_samples:
            raise ValueError(
                f"{self.meta.edf_id}: signal length {len(self.signal)} != "
                f"meta.n_samples {self.meta.n_samples}"
            )

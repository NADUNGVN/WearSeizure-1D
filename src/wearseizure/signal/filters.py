"""Causal signal conditioning shared by training and the integer reference.

Research Decision Memo 2.2 ("Causal mismatch"): the original paper's offline
zero-phase filtering / whole-file normalization cannot be deployed streaming.
Training and the RTL/integer reference must share the *same* causal filter,
the *same* rounding, and the *same* scale. `CausalBandpass` is therefore the
one implementation used everywhere a band-pass is needed, and its per-EDF
state is always reset at file boundaries (never carried across recordings).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi

DEFAULT_LOW_HZ = 1.0
DEFAULT_HIGH_HZ = 30.0
DEFAULT_ORDER = 4


@dataclass(frozen=True)
class FilterState:
    zi: np.ndarray


class CausalBandpass:
    """One-pass (causal) Butterworth band-pass, IIR, applied with `lfilter`.

    A true zero-phase (`filtfilt`) design is non-causal by construction and is
    intentionally never used here, even for baseline reproduction -- see memo
    2.2. Per-EDF state must be reset (`initial_state`) at the start of every
    recording; never carry `zi` across files.
    """

    def __init__(
        self,
        low_hz: float = DEFAULT_LOW_HZ,
        high_hz: float = DEFAULT_HIGH_HZ,
        fs_hz: float = 256.0,
        order: int = DEFAULT_ORDER,
    ) -> None:
        nyq = fs_hz / 2.0
        if not (0 < low_hz < high_hz < nyq):
            raise ValueError(f"invalid band [{low_hz}, {high_hz}] for fs={fs_hz}")
        self.low_hz = low_hz
        self.high_hz = high_hz
        self.fs_hz = fs_hz
        self.order = order
        self.b, self.a = butter(order, [low_hz / nyq, high_hz / nyq], btype="band")

    def initial_state(self, first_sample: float = 0.0) -> FilterState:
        zi = lfilter_zi(self.b, self.a) * first_sample
        return FilterState(zi=zi)

    def apply(self, x: np.ndarray, state: FilterState | None = None) -> tuple[np.ndarray, FilterState]:
        if x.ndim != 1:
            raise ValueError(f"CausalBandpass.apply expects 1-D input, got shape {x.shape}")
        if state is None:
            state = self.initial_state(first_sample=float(x[0]) if len(x) else 0.0)
        y, zf = lfilter(self.b, self.a, x, zi=state.zi)
        return y, FilterState(zi=zf)

    def apply_full_reset(self, x: np.ndarray) -> np.ndarray:
        """Convenience for a full EDF: reset state at t=0, filter causally to the end."""
        y, _ = self.apply(x, state=None)
        return y

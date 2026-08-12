"""Causal exponential moving average over per-window scores (memo 4.5: alpha=1/8)."""
from __future__ import annotations

import numpy as np


def ema_smooth(scores: np.ndarray, alpha: float) -> np.ndarray:
    if not (0 < alpha <= 1):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    if len(scores) == 0:
        return np.asarray(scores, dtype=np.float64)
    out = np.empty(len(scores), dtype=np.float64)
    out[0] = scores[0]
    for i in range(1, len(scores)):
        out[i] = alpha * scores[i] + (1 - alpha) * out[i - 1]
    return out

"""Affine normalization fit on train only, applied causally (stateless, memoryless).

The model's Input stage (Table 4) is "causal band-pass + affine normalize":
after the causal filter, normalization is a fixed per-subject affine map
(scale, bias) estimated once from train-partition statistics and then frozen.
Because it is memoryless (no running state), applying it sample-by-sample in
a stream is trivially causal and bit-identical to applying it to a full array.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AffineNormalizer:
    scale: float
    bias: float

    def apply(self, x: np.ndarray) -> np.ndarray:
        return (x - self.bias) * self.scale

    def to_dict(self) -> dict:
        return {"scale": self.scale, "bias": self.bias}


def fit_affine_normalizer(train_signals: list[np.ndarray], robust: bool = True) -> AffineNormalizer:
    """Fit bias/scale from train-partition filtered signals only.

    `robust=True` uses median/MAD (resistant to the large-amplitude artifacts
    called out in the memo's synthetic-data design) instead of mean/std.
    """
    concat = np.concatenate([s.ravel() for s in train_signals])
    if concat.size == 0:
        raise ValueError("fit_affine_normalizer: no samples provided")
    if robust:
        bias = float(np.median(concat))
        # In place, deliberately. `np.median(np.abs(concat - bias))` allocates
        # TWO more arrays the size of the whole train partition, on top of
        # `concat` itself -- about 34 GiB of float64 temporaries for the
        # lever-L5 corpus, which is what OOM-killed the first Phase 3 attempt
        # (anon-rss 72.6 GiB per process against 188 GiB shared by three).
        # Writing through `out=` gives bit-identical values with none of them.
        np.subtract(concat, bias, out=concat)
        np.abs(concat, out=concat)
        mad = float(np.median(concat))
        # `concat` now holds |x - bias|, so the old `np.std(concat)` fallback no
        # longer sees the original samples. It is unreachable in practice --
        # mad == 0 requires more than half the samples to equal the median
        # exactly, i.e. a flat-lined recording -- and a unit scale is the right
        # answer for a constant signal anyway.
        spread = mad * 1.4826 if mad > 0 else 1.0
    else:
        bias = float(np.mean(concat))
        spread = float(np.std(concat)) or 1.0
    return AffineNormalizer(scale=1.0 / spread, bias=bias)

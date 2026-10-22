"""Class-balanced sampling -- train partition only (memo 5.1 step 3: class
sampler is fit only on train/validation, never on test).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

from wearseizure.data.dataset import WearSeizureWindowDataset


def onset_proximity_boost(
    dataset: WearSeizureWindowDataset, within_s: float, factor: float
) -> np.ndarray:
    """Per-window multiplier emphasising the first seconds of each seizure.

    Measured on the handover vectors: on `chb15_40` every window the detector
    gets wrong is a false negative in the first eighteen seconds after onset,
    and there is not one false positive. The detector is not misclassifying,
    it is reacting late -- about fifteen seconds of that delay is the model
    rather than the four-second window floor or the run-length requirement.

    Those early-ictal windows are also the hardest examples in the set, because
    three quarters of a window straddling the onset is pre-ictal signal. Under
    plain class-balanced sampling they are a small minority of the ictal class
    and contribute correspondingly little gradient, so the model is free to
    learn a confident mid-seizure detector and pay almost nothing for being
    slow. This raises their sampling probability so that being slow costs
    something.

    `within_s` is measured from the onset to the window's END, which is the
    causal instant the decision is available. A window ending exactly at the
    onset contains no ictal signal and is not boosted.

    Returns ones when the boost is off, so the caller's arithmetic is
    unchanged rather than merely equivalent.
    """
    n = len(dataset.windows)
    boost = np.ones(n, dtype=np.float64)
    if within_s <= 0.0 or factor == 1.0:
        return boost
    if factor <= 0.0:
        raise ValueError(f"onset boost factor must be > 0, got {factor}")

    # Onsets per EDF, read from the record metadata rather than from labels:
    # the label says "this window touches a seizure", which cannot distinguish
    # the first second of one from the ninetieth.
    onsets: dict[str, np.ndarray] = {}
    for edf_id, record in dataset.records.items():
        times = [e.onset_sec for e in record.meta.seizure_events]
        onsets[edf_id] = np.asarray(sorted(times), dtype=np.float64)

    for i, w in enumerate(dataset.windows):
        o = onsets.get(w.edf_id)
        if o is None or not len(o):
            continue
        # Seconds from the most recent onset at or before this window's end.
        prior = o[o < w.end_sec]
        if not len(prior):
            continue
        if w.end_sec - prior[-1] <= within_s:
            boost[i] = factor
    return boost


def make_class_balanced_sampler(
    dataset: WearSeizureWindowDataset,
    onset_within_s: float = 0.0,
    onset_factor: float = 1.0,
) -> WeightedRandomSampler:
    labels = np.array([w.label for w in dataset.windows])
    class_counts = np.bincount(labels, minlength=2).astype(np.float64)
    class_weights = 1.0 / np.clip(class_counts, 1.0, None)
    sample_weights = class_weights[labels]
    # Applied on top of the class weights, not instead of them: the point is to
    # redistribute probability WITHIN the ictal class, not to unbalance the two.
    sample_weights = sample_weights * onset_proximity_boost(
        dataset, onset_within_s, onset_factor)
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(labels),
        replacement=True,
    )

"""Class-balanced sampling -- train partition only (memo 5.1 step 3: class
sampler is fit only on train/validation, never on test).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

from wearseizure.data.dataset import WearSeizureWindowDataset


def make_class_balanced_sampler(dataset: WearSeizureWindowDataset) -> WeightedRandomSampler:
    labels = np.array([w.label for w in dataset.windows])
    class_counts = np.bincount(labels, minlength=2).astype(np.float64)
    class_weights = 1.0 / np.clip(class_counts, 1.0, None)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(labels),
        replacement=True,
    )

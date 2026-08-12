"""Deterministic seeding shared by data generation, splitting, and training."""
from __future__ import annotations

import hashlib
import random

import numpy as np


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def rng_for(*parts: object, base_seed: int) -> np.random.Generator:
    """Derive a reproducible RNG from a base seed plus arbitrary identifying parts.

    Used so that, e.g., every (subject_id, seed) pair gets its own stable stream
    without ever reusing the same sequence across subjects or folds.
    """
    key = "|".join(str(p) for p in (*parts, base_seed))
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    derived = int.from_bytes(digest[:8], "big") % (2**32 - 1)
    return np.random.default_rng(derived)
